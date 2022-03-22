#!/bin/bash
cd ..
nohup python3 train.py --gpu 0,1,2,3 --batch_size 32 > ./log/exp10_128_2_train.txt