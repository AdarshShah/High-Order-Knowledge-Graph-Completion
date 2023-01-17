#!/bin/sh
for n in {1..10};
do
    python train.py -e t_3 --dataset cat_edge_cooking --num_classes 20 --iter 20000 --reset_model
    python test.py -e t_3 --dataset cat_edge_cooking --num_classes 20 --iter 5000
    python test.py -e t_3 --dataset cat_edge_cooking --num_classes 20 --iter 5000 --dim 0
    python test.py -e t_3 --dataset cat_edge_cooking --num_classes 20 --iter 5000 --dim 1
    python test.py -e t_3 --dataset cat_edge_cooking --num_classes 20 --iter 5000 --dim 2
    python test.py -e t_3 --dataset cat_edge_cooking --num_classes 20 --iter 5000 --dim 3
done