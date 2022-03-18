#!/bin/bash
cd ..
nohup python train.py --gpu 0,1,2,3,4,5,6,7 --batch_size 120 > ./log/exp9_train.txt