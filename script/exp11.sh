#!/bin/bash
cd ..
nohup python train.py --gpu 0,1,2,3 --batch_size 28 > ./log/exp11_train.txt