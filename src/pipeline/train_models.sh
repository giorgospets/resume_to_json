#!/bin/bash

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

cd ${root_dir}

python ./src/scripts/model_lora_finetune.py

python ./src/scripts/model_full_finetune.py
