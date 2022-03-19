#!/bin/bash
cd ..
nohup python train.py --gpu 4,5,6,7 --batch_size 112 > ./log/exp10_128_2_train.txt