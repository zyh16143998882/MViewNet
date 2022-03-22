#!/bin/bash
cd ..
nohup python3 train.py --gpu 0,1,2,3 --batch_size 52 > ./log/exp10_128_1_gt.txt