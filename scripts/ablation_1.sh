#!/bin/sh
python train.py -e ablation_1 --dataset cat_edge_cooking --num_classes 20 --iter 5000 --reset_model
python test.py -e ablation_1 --dataset cat_edge_cooking --num_classes 20 --iter 5000
for n in {1..9};
do
    python train.py -e ablation_1 --dataset cat_edge_cooking --num_classes 20 --iter 5000
    python test.py -e ablation_1 --dataset cat_edge_cooking --num_classes 20 --iter 5000
done