#!/bin/bash
cd ..
nohup python3 train.py --gpu 0,1,2,3 --batch_size 8 > ./log/exp_mviewnet_train.txt