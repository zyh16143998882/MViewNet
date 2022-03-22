#!/bin/bash
cd ..
nohup python3 train.py --gpu 0,1,2,3,4,5,6,7 --batch_size 232 > ./log/exp11_train.txt