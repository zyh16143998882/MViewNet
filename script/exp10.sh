#!/bin/bash
cd ..
nohup python train.py --gpu 0,1,2,3 --batch_size 128 > ./log/exp10_128_1_train.txt