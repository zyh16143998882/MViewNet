#!/bin/bash
cd ..
nohup python train.py --gpu 0,1,2,3 --batch_size 20 > ./log/exp8_train.txt