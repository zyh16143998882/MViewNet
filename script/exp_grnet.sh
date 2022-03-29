#!/bin/bash
cd ..
nohup python3 train.py --gpu 0,1,2,3 --batch_size 356 > ./log/exp_grnet.txt