#!/bin/sh
for n in {1..10}
do
python train.py --dataset cat_edge_walmart_trips --num_classes 15 --iter 10000
python test.py --dataset cat_edge_walmart_trips --num_classes 15 --iter 1000
done