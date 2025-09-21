#!/bin/bash

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

cd ${root_dir}

python src/scripts/model_test.py --model_name lora_finetuned_gemma-3-4b-it-4bit_epoch_1.0 --lora

python src/scripts/model_test.py --model_name lora_finetuned_gemma-3-1b-it-4bit_epoch_4.0 --lora

python ./src/scripts/model_test.py --model_name full_finetuned_gemma-3-270m --checkpoint 150
