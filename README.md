# Towards Calibrated Robust Fine-Tuning of Vision-Language Models
Changdae Oh*, Hyesu Lim*, Mijoo Kim, Dongyoon Han, Sangdoo Yun, Jaegul Choo, Alexander Hauptmann, Zhi-Qi Cheng^, Kyungwoo Song^

[arXiv](https://arxiv.org/abs/2311.01723)

<br/>
<br/>
<br/>


## Setting up conda env
```bash
conda env create -f carot_env.yaml
conda activate carot
mkdir checkpoints
```

### Add directory to PYTHONPATH:

```bash
cd CaRot
export PYTHONPATH="$PYTHONPATH:$PWD"
```


### Script to reproduce the main result
* Refer to the DATA.md for the ImageNet directory strucutre.
* Refer example_scripts for other methods.

```bash
ln -s PATH_TO_YOUR_ILSVRC2012_DATASET ./datasets/data/ILSVRC2012

python datacreation_scripts/imagenet_csv_creator.py

OC=0.2
SD=1.5

python src/main.py \
--train-dataset=ImageNet --epochs=10 --lr 1e-5 --wd 0.1 --batch-size 512 \
--model=ViT-B/16 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
--template=openai_imagenet_template  --save=./checkpoints/ \
--data-location=./datasets/data/ --ft_data="./datasets/csv/imagenet.csv" \
--csv-img-key filepath --csv-caption-key title --exp_name ImageNet/carot \
--cross_fnorm 0.05 --l_orth_wv $OC --distil_coef $SD \
--wb_project "YOUR-PROJECT-NAME" --method carot
```

<br/>

---

<br/>

### Acknowledge
This repository is built on top of the [FLYP](https://github.com/locuslab/FLYP) project. We thank the authors for sharing the source and their work itself.
