import os
import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help=
        "Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
        " Note that same model used for all datasets, so much have same classnames"
        "for zero shot.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help=
        "Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--alpha",
        default=[0.5],
        nargs='*',
        type=float,
        help=
        ('Interpolation coefficient for ensembling. '
         'Users should specify N-1 values, where N is the number of '
         'models being ensembled. The specified numbers should sum to '
         'less than 1. Note that the order of these values matter, and '
         'should be the same as the order of the classifiers being ensembled.'
         ))
    
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.")
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument("--lr",
                        type=float,
                        default=0.00001,
                        help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")

    parser.add_argument("--ls",
                        type=float,
                        default=0.0,
                        help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--vis_calibration",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help=
        "Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=
        "Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help=
        "Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8,
        help="TODO",
    )

    parser.add_argument(
        "--ft_data",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )

    parser.add_argument('--ce_ablation', action=argparse.BooleanOptionalAction)


    parser.add_argument("--dataset-type",
                        choices=["webdataset", "csv", "auto"],
                        default="auto",
                        help="Which type of dataset to process.")

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help=
        "Number of samples in dataset. Required for webdataset if not available in info file.",
    )

    parser.add_argument("--k",
                        type=int,
                        default=None,
                        help="k for few shot ImageNet")
                        
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Default random seed.")

    parser.add_argument("--workers",
                        type=int,
                        default=16,
                        help="Number of dataloader workers per GPU.")

    parser.add_argument("--csv-separator",
                        type=str,
                        default="\t",
                        help="For csv-like datasets, which separator to use.")
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths.")
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions.")


    parser.add_argument(
        "--clip_load",
        type=str,
        default=None,
        help="Load finetuned clip",
    )

    parser.add_argument(
        "--wise_save",
        type=str,
        default=None,
        help="Save path for wiseft results",
    )

    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Repeated run number",
    )

    parser.add_argument("--get_labeled_csv",
                        default=False,
                        action="store_true",
                        help="get labels from csv.")


    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="minimum LR for cosine scheduler",
    )
    #! lp-ft --------------------------
    parser.add_argument(
        "--head_path",
        type=str,
        default='',
        help="pre-trained head for lp-ft",
    )

   
    #! carot --------------------------
    parser.add_argument(
        "--distil_coef",
        type=float,
        default=0.0,
        help="coefficient for self-distillation loss",
    )

    parser.add_argument(
        "--ema_up_freq",
        type=int,
        default=500,
        help="required iterations for EMA teacher update",
    )

    parser.add_argument(
        "--m_sche_src",
        type=float,
        default=0.05,
        help="EMA teacher evolving schedule (src)",
    )

    parser.add_argument(
        "--m_sche_tar",
        type=float,
        default=0.9,
        help="EMA teacher evolving schedule (tar)",
    )

    parser.add_argument(
        "--m_warm_up",
        type=float,
        default=0.2,
        help="EMA teacher evolving schedule (warmup ratio)",
    )
    parser.add_argument(
        "--cross_fnorm",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--l_orth_wv",
        type=float,
        default=0.0,
    )
    #! ---------------
    parser.add_argument(
        "--wb_project",
        type=str,
        default="",
        help="weight and bias project name",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="",
        help="zs / ft / flyp / lpft / carot ...",
    )

    parser.add_argument(
        "--use_fp16",
        type=int,
        default=1,
        help="mixed precision training flag",
    )

    parser.add_argument(
        "--temperature_scale",
        type=float,
        default=0.0,
        help="temperature scaling",
    )
    parser.add_argument(
        "--full_eval",
        type=int,
        default=0,
        help="temperature scaling",
    )

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
