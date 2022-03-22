#!/bin/bash
cd ..
nohup python3 train_partial.py --gpu 0,1,2,3 --batch_size 30 --pretrain "./checkpoint/pretrain/ckpt-best.pth" > ./log/exp_pcf_train.txt