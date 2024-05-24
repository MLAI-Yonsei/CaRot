#!/bin/bash

cd ..

lr=1e-5
wd=0.1
bs=512
ts=0.0
method=carot
fp16=1

for sd in 1.5
do
for oc in 0.2
do

CUDA_VISIBLE_DEVICES='0,1' python src/main.py \
--train-dataset=ImageNet --epochs=10 --lr ${lr} --wd ${wd} --batch-size $bs \
--model=ViT-B/16 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
--template=openai_imagenet_template  --save=./checkpoints/ \
--data-location=./datasets/data/ --ft_data="./datasets/csv/imagenet.csv" \
--csv-img-key filepath --csv-caption-key title --exp_name ImageNet/${method} --cross_fnorm=0.05 \
--distil_coef $sd --l_orth_wv $oc \
--wb_project "YOUR-PROJECT-NAME" --method $method

done
done