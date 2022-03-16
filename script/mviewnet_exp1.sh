#!/bin/bash
cd ..
nohup python train.py --gpu 0,1,2,3 --batch_size 52 > ./log/exp_mviewnet_train.txt