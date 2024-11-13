#!/bin/bash

# 训练脚本路径
SCRIPT_TO_RUN="/home/sp/projects/git_update/OpenGait/train.sh"

# 检查是否有正在运行的训练任务 191357 ，nvidia-smi | grep "python"得出
while nvidia-smi | grep 191357; do
    sleep 240 # 防止频繁检查消耗太多资源
done

# 如果没有运行的任务，则启动新的训练脚本
$SCRIPT_TO_RUN