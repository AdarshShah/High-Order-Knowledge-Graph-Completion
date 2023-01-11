#!/bin/sh
for n in {1..10}
do
python train.py --dataset cat_edge_MAG_10 --num_classes 10 --iter 10000
python test.py --dataset cat_edge_MAG_10 --num_classes 10 --iter 1000
done