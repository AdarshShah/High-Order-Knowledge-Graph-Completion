#!/bin/sh
for n in {1..10};
do
    python train.py -e t_3 --dataset cat_edge_algebra_questions --num_classes 20 --iter 20000 --reset_model
    python test.py -e t_3 --dataset cat_edge_algebra_questions --num_classes 20 --iter 5000
done
