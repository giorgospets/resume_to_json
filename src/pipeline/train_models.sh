#!/bin/bash

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

cd ${root_dir}

python ./src/scripts/model_lora_finetune_4b.py --num_epochs 1

python ./src/scripts/model_lora_finetune_1b.py --num_epochs 4

python ./src/scripts/model_full_finetune.py
