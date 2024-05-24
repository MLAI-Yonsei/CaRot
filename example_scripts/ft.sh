#!/bin/bash

cd ..

lr=3e-5
wd=0.1
bs=512
method=ft

CUDA_VISIBLE_DEVICES='0' python src/main.py \
--train-dataset=ImageNet --epochs=10 --lr ${lr} --wd ${wd} --batch-size $bs \
--model=ViT-B/16 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
--template=openai_imagenet_template  --save=./checkpoints/ \
--data-location=./datasets/data/ --ft_data="./datasets/csv/imagenet.csv" \
--csv-img-key filepath --csv-caption-key title --exp_name ImageNet/${method} \
--wb_project "YOUR-PROJECT-NAME" --method $method