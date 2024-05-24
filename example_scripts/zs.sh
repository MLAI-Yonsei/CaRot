#!/bin/bash

cd ..

method=zs


CUDA_VISIBLE_DEVICES='0' python src/main.py \
--train-dataset=ImageNet --epochs=0  \
--model=ViT-B/16 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
--template=openai_imagenet_template  --save=./checkpoints/ \
--data-location=./datasets/data/ --ft_data="./datasets/csv/imagenet.csv" \
--csv-img-key filepath --csv-caption-key title --exp_name ImageNet/${method} \
--wb_project "YOUR-PROJECT-NAME" --method $method