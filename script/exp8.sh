#!/bin/bash
cd ..
nohup python3 train.py --gpu 0,1,2,3 --batch_size 20 > ./log/exp_dec8_train.txt