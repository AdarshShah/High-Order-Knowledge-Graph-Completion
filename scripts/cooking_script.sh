#!/bin/sh
for n in {1..10};
do
    python train.py -e t_1 --dataset cat_edge_cooking --num_classes 20 --iter 10000 --reset_model
    python test.py -e t_1 --dataset cat_edge_cooking --num_classes 20 --iter 1000
    python test.py -e t_1 --dataset cat_edge_cooking --num_classes 20 --iter 1000 --dim 0
    python test.py -e t_1 --dataset cat_edge_cooking --num_classes 20 --iter 1000 --dim 1
    python test.py -e t_1 --dataset cat_edge_cooking --num_classes 20 --iter 1000 --dim 2
    python test.py -e t_1 --dataset cat_edge_cooking --num_classes 20 --iter 1000 --dim 3
done