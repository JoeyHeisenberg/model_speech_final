#!/bin/bash

for epoch in {16..300}
do
    echo "epoch index: $epoch"
    if [ $epoch -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=1 python libri_train.py --mode=train --num_epochs=$epoch|| break
    else
        CUDA_VISIBLE_DEVICES=1 python libri_train.py --batch_size=32 --mode=train --keep=True --num_epochs=$epoch|| break
        CUDA_VISIBLE_DEVICES=1 python libri_train.py --mode=dev --keep=True || break # 使用”||”时，command1执行成功后command2 不执行，否则去执行command2.
        CUDA_VISIBLE_DEVICES=1 python libri_train.py --mode=test --keep=True || break
    fi
done
