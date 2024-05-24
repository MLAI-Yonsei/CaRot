from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
import tqdm
import wandb

import torch
from torch.nn import functional as F
import pandas as pd
import clip.clip as clip
from clip.loss import ClipLoss

from src.args import parse_arguments
from src.datasets_.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits, clip_img_preprocessing, attack_pgd
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets_.laion import get_data
import src.datasets_ as datasets


def finetune(args, image_classifier):
    #assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."
    
    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        image_enc = image_classifier.image_encoder
        print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
        image_classifier.process_images = True
        print_every = 100
    
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)
    print(f"Num batches is {num_batches}")

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    if args.clip_load is not None:
        model = model.load(args.clip_load)
        if args.head_path:
            load_epoch = 5 if args.train_dataset == 'ImageNet' else 10
            head_path = os.path.join(args.head_path, f'checkpoint_lp_{load_epoch}.pt')
            model.classification_head = model.classification_head.load(head_path)

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    
    stats = []
    prev_num_logits = 0
    labels_ = {}
    
    if args.epochs == 0:
        epoch = 0
        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        args.current_epoch = epoch
        
        if args.method in ['ft', 'ft_ema','ftls', 'lpft','lp']:
            if args.freeze_encoder:
                image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
            else:
                image_classifier = model.module
        else:
            pass
        eval_results = evaluate(image_classifier, args, train_stats=epoch_stats)
        wandb.log({k:v for k, v in epoch_stats.items()})
        exit()
    #!
    
    tot_loss = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_stats = {}
        epoch_stats['epoch'] = epoch

        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=image_enc)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                logits = model(inputs)
                loss = loss_fn(logits, labels)
            tot_loss += loss.item()

            if fp16_scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            batch_time = time.time() - start_time

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

    # Evaluate
    args.current_epoch = epoch
    if args.freeze_encoder:
        image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
    else:
        image_classifier = model.module

    eval_results = evaluate(image_classifier, args, train_stats=epoch_stats)

    # Saving model
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
        print('Saving model to', model_path)
        image_classifier.save(model_path)
        torch.save(image_classifier, os.path.join(args.save, f'_checkpoint_{epoch+1}.pt'))
        optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
        torch.save(optimizer.state_dict(), optim_path)
        if args.method == 'lp':
            haed_path = os.path.join(args.save, f'checkpoint_lp_{epoch+1}.pt')
            image_classifier.classification_head.save(haed_path)

    print(f"Avg ID Loss : {tot_loss / len(data_loader):.4f}")
    epoch_stats['Avg ID Loss'] = round(tot_loss / len(data_loader), 4)
    stats.append(epoch_stats)
    stats_df = pd.DataFrame(stats)
    log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs(log_dir, exist_ok=True)
    stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')

    if args.save is not None:
        return model_path