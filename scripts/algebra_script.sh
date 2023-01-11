#!/bin/sh
for n in {1..10};
do
    python train.py --dataset cat_edge_algebra_questions --num_classes 32 --iter 10000
    python test.py --dataset cat_edge_algebra_questions --num_classes 32 --iter 1000
    python test.py --dataset cat_edge_algebra_questions --num_classes 32 --iter 1000 --dim 0
    python test.py --dataset cat_edge_algebra_questions --num_classes 32 --iter 1000 --dim 1
    python test.py --dataset cat_edge_algebra_questions --num_classes 32 --iter 1000 --dim 2
    python test.py --dataset cat_edge_algebra_questions --num_classes 32 --iter 1000 --dim 3
done
