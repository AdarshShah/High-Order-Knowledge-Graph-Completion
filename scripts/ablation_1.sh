#!/bin/sh
python train.py -e ablation_1 --dataset cat_edge_cooking --num_classes 20 --iter 1000 --reset_model
python test.py -e ablation_1 --dataset cat_edge_cooking --num_classes 20 --iter 1000
for n in {1..9};
do
    python train.py -e ablation_1 --dataset cat_edge_cooking --num_classes 20 --iter 1000
    python test.py -e ablation_1 --dataset cat_edge_cooking --num_classes 20 --iter 1000
done