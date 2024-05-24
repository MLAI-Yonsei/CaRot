from ast import arg
import os
import numpy as np
import torch
from src.models.eval import evaluate
from src.models.ft_loss import finetune
from src.models.carot_loss import carot_loss
from src.models.ce_ablation import ce_ablation
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
import logging
import random

import wandb
import glob


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)


def main(args):
    set_seed(args.run)
    if args.wb_project:
        wandb_args = {"project": args.wb_project}
        wandb_args["name"] = args.method if args.method else None
        wandb.init(**wandb_args, config=vars(args), save_code=False)

    def modelname_generator(argmodel):
        if argmodel == 'ViT-B/32': mn = 'VITB32'
        if argmodel == 'ViT-B/16': mn = 'VITB16'
        if argmodel == 'ViT-L/14': mn = 'VITL14'
        if argmodel == 'ViT-L/14@336px': mn = 'VITL14px'
        else: mn = argmodel
        return mn
        
    mod_flag = modelname_generator(args.model)

    os.makedirs(args.save + args.exp_name, exist_ok=True)
    if args.adv_training:
        args.save = args.save + args.exp_name + "/" + f"{mod_flag}" + '_ep' + str(args.epochs) + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_D" + str(args.distil_coef) + "_OC" +str(args.l_orth_wv) + "_run" + str(args.run)
        os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
        logging_path = "expt_logs/" + args.exp_name + "/" + f"{mod_flag}" + '_ep' + str(args.epochs) + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_D" + str(args.distil_coef) + "_OC" +str(args.l_orth_wv) + "_run" + str(args.run)
    else:
        args.save = args.save + args.exp_name + "/" + f"{mod_flag}" + '_ep' + str(args.epochs) + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_D" + str(args.distil_coef) + "_OC" +str(args.l_orth_wv) + "_run" + str(args.run)
        os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
        logging_path = "expt_logs/" + args.exp_name + "/" + f"{mod_flag}" + '_ep' + str(args.epochs) + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_D" + str(args.distil_coef) + "_OC" +str(args.l_orth_wv) + "_run" + str(args.run)

    
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(
        filename=log_filename, format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    assert args.save is not None, "Please provide a path to store models"
    #############################################################################

    # Initialize the CLIP encoder
    clip_encoder = CLIPEncoder(args, keep_lang=True)
    if args.head_path:
        classification_head = ClassificationHead.load(args.head_path)
    else:
        if args.method == "lp":
            outdim = 0
            if args.train_dataset == 'ImageNet':           outdim = 1000
            elif args.train_dataset == 'IWildCamID':       outdim = 182
            elif args.train_dataset == 'FMOWID':           outdim = 62
            elif args.train_dataset == 'sst2Val':          outdim = 2
            elif args.train_dataset == 'PatchCamelyonVal': outdim = 2
            elif args.train_dataset == 'Caltech101Val':    outdim = 101
            elif args.train_dataset == 'StanfordCarsVal':  outdim = 196
            elif args.train_dataset == 'Flowers102Val':    outdim = 102
        
            if   args.model == 'RN50':     indim = 1024
            elif args.model == 'RN50x4':   indim = 640
            elif args.model == 'ViT-L/14': indim = 768
            else:  indim = 512
            classification_head = ClassificationHead(normalize=None, weights=None, shape=[indim, outdim])
        else:
            classification_head = get_zeroshot_classifier(args, clip_encoder.model)

    logger.info(args)

    if args.method in ['lp','ft','lpft']:
        delattr(clip_encoder.model, 'transformer')
        image_clf = ImageClassifier(clip_encoder, classification_head, process_images=False)
        finetuned_checkpoint = finetune(args, image_clf)
    else:
        finetuned_checkpoint = carot_loss(args, clip_encoder,
                                            classification_head, logger)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
