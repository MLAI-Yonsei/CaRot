from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
import tqdm
import wandb
import pdb

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


def carot_loss(args, clip_encoder, classification_head, logger):
    assert args.train_dataset is not None, "Please provide a training dataset."

    logger.info("Fine-tuning Using carot Loss")
    model = clip_encoder
    input_key = "images"
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True
    print_every = 100

    dataset_class = getattr(datasets, args.train_dataset)
    print(f"Training dataset {args.train_dataset}")

    dataset = dataset_class(
        preprocess_fn, location=args.data_location, batch_size=args.batch_size
    )

    img_text_data = get_data(
        args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess), epoch=0
    )
    assert len(img_text_data), "At least one train or eval dataset must be specified."
    ft_dataloader = img_text_data["train_ft"].dataloader
    ft_iterator = iter(ft_dataloader)
    num_batches = len(dataset.train_loader)
    print(f"Num batches is {num_batches}")

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    if args.clip_load is not None:
        model = model.load(args.clip_load)

    if args.distil_coef:
        import copy

        teacher_enc = copy.deepcopy(model).cuda()

    model = model.cuda()

    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    logger.info("Using devices" + str(devices))

    model = torch.nn.DataParallel(model, device_ids=devices)

    if args.distil_coef:
        teacher_enc.load_state_dict(model.module.state_dict())

    classification_head = torch.nn.DataParallel(classification_head, device_ids=devices)
    classification_head.train()
    model.train()

    clip_loss_fn = ClipLoss(
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=0,
        world_size=1,
        use_horovod=False,
        ls=args.ls,
    )

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer, args.lr, args.warmup_length, args.epochs * num_batches, args.min_lr
    )

    stats = []
    prev_num_logits = 0
    labels_ = {}
    #! inference flag
    if args.epochs == 0:
        epoch = 0
        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats["epoch"] = epoch
        args.current_epoch = epoch
        
        print("Start evaluation")
        classification_head_new = get_zeroshot_classifier(args, model.module.model)
        classification_head_new = classification_head_new.cuda()
        eval_results = evaluate(
            model, args, classification_head_new, epoch_stats, logger
        )
        wandb.log({k: v for k, v in epoch_stats.items()})
        exit()

    for epoch in range(0, args.epochs):
        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats["epoch"] = epoch
        id_carot_loss_sum = 0
        model.train()
        model = model.cuda()
        classification_head.train()

        for i in range(num_batches):
            start_time = time.time()
            step = i + epoch * num_batches
            if epoch != -1:
                scheduler(step)
            optimizer.zero_grad()

            try:
                ft_batch = next(ft_iterator)
            except StopIteration:
                ft_iterator = iter(
                    ft_dataloader
                )  # If ft_iterator is all used, re-initialize it
                ft_batch = next(ft_iterator)
            
            ft_image, ft_text = ft_batch
            ft_image, ft_text = ft_image.cuda(), ft_text.cuda()
            
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                ft_image_features, ft_text_features, logit_scale2 = model(
                    ft_image, ft_text
                )

                lscale = logit_scale2 if len(devices) == 1 else logit_scale2[0]

                ft_clip_loss, logits_per_image, logits_per_text = clip_loss_fn(
                    ft_image_features, ft_text_features, lscale
                )

                #* d-rank SVD approximation
                if args.cross_fnorm:
                    if args.model[:3] != 'ViT':
                        cov_vl = model.module.model.visual.attnpool.c_proj.weight.T @ model.module.model.text_projection.T
                    else:
                        cov_vl = model.module.model.visual.proj @ model.module.model.text_projection.T
                    ft_clip_loss += args.cross_fnorm * torch.linalg.norm(cov_vl, ord='fro')

                #* orthogonality constraint
                if args.l_orth_wv:
                    if args.model[:3] != 'ViT':
                        covv = model.module.model.visual.attnpool.c_proj.weight.T @ model.module.model.visual.attnpool.c_proj
                    else:
                        covv = model.module.model.visual.proj.T @ model.module.model.visual.proj
                    ft_clip_loss += args.l_orth_wv * ((covv - torch.eye(covv.shape[0], device=covv.device))**2).sum()**(1/2)

            #! self-distillation flag
            dist_loss, m = torch.tensor(0), 0.0
            if args.distil_coef:
                if step > 0:
                    with torch.cuda.amp.autocast(fp16_scaler is not None):
                        with torch.no_grad():
                            (
                                ft_image_features_t,
                                ft_text_features_t,
                                logit_scale_t,
                            ) = teacher_enc(ft_image, ft_text)

                            logits_per_image_t = (
                                logit_scale_t
                                * ft_image_features_t
                                @ ft_text_features_t.T
                            )
                            logits_per_text_t = (
                                logit_scale_t
                                * ft_text_features_t
                                @ ft_image_features_t.T
                            )
                        
                        dist_loss = -torch.sum(
                            F.softmax(logits_per_image_t, dim=1)
                            * torch.log(F.softmax(logits_per_image, dim=1))
                            + F.softmax(logits_per_text_t, dim=1)
                            * torch.log(F.softmax(logits_per_text, dim=1)),
                            dim=1
                        ).mean()
                        
                        ft_clip_loss += args.distil_coef * dist_loss

            if fp16_scaler is None:
                ft_clip_loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(ft_clip_loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            #! self-distillation
            if args.distil_coef:
                if args.ema_up_freq <= 0:
                    pass
                else:
                    if ((step % args.ema_up_freq) == 0) or (
                        step == num_batches * args.epochs
                    ):
                        if step < num_batches * args.epochs * args.m_warm_up:
                            m = (
                                (args.m_sche_tar - args.m_sche_src)
                                / (num_batches * args.epochs * args.m_warm_up)
                            ) * step + args.m_sche_src
                        else:
                            m = args.m_sche_tar
                        
                        for param_q, param_k in zip(
                            model.module.parameters(), teacher_enc.parameters()
                        ):
                            param_k.data.mul_(m).add_(
                                (1 - m) * param_q.detach().data
                            )

            id_carot_loss_sum += ft_clip_loss.item()

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                logger.info(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                    f"ID Loss: {ft_clip_loss.item():.4f}"
                )

    id_carot_loss_avg = id_carot_loss_sum / num_batches

    # Evaluate
    args.current_epoch = epoch
    classification_head_new = get_zeroshot_classifier(args, model.module.model)
    classification_head_new = classification_head_new.cuda()

    eval_results = evaluate(model, args, classification_head_new, epoch_stats, logger)

    # Saving model
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, f"checkpoint_{epoch+1}.pt")
        logger.info("Saving model to" + str(model_path))
        model.module.save(model_path)

        #! save the EMA teacher
        ema_model_path = os.path.join(args.save, f"checkpoint_{epoch+1}_EMA.pt")
        logger.info("Saving model to" + str(ema_model_path))
        try:
            teacher_enc.save(ema_model_path)
        except:
            print("============================")
            print("error occurred during EMA model saving")
            print("============================")

        optim_path = os.path.join(args.save, f"optim_{epoch+1}.pt")
        torch.save(optimizer.state_dict(), optim_path)

    logger.info(f"Avg ID carot Loss : {id_carot_loss_avg:.4f}")
    epoch_stats["Avg ID carot Loss"] = round(id_carot_loss_avg, 4)
    stats.append(epoch_stats)
    stats_df = pd.DataFrame(stats)
    log_dir = (
        "expt_logs/"
        + args.exp_name
        + "/"
        + "_BS"
        + str(args.batch_size)
        + "_WD"
        + str(args.wd)
        + "_LR"
        + str(args.lr)
        + "_run"
        + str(args.run)
    )
    os.makedirs(log_dir, exist_ok=True)
    stats_df.to_csv(log_dir + "/stats.tsv", sep="\t")

    #! wandb logging
    wandb.log({k: v for k, v in epoch_stats.items()})

    if args.save is not None:
        return model_path
