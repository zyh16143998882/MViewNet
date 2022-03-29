#!/bin/bash
cd ..
nohup python train.py --gpu 0,1,2,3 --batch_size 50 > ./log/mview_worefine.txt

nohup python train.py --gpu 0,1,2,3 --batch_size 50 --refine > ./log/mview_refine.txt