#!/bin/bash

# 使用 nvidia-smi 检查显卡的使用情况
# 查询显卡的索引和显存使用情况
empty_gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F ', ' '{if ($2 < 1024) print $1}' | head -n 1)

if [ -z "$empty_gpu" ]; then
    echo "没有空闲的显卡。"
else
    echo "第一个空闲的显卡是: GPU $empty_gpu"
fi
