#!/bin/bash

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

cd ${root_dir}

python src/scripts/model_test.py --model_name lora_finetuned_gemma-3-4b-it-4bit_epoch_1 --lora

python src/scripts/model_test.py --model_name lora_finetuned_gemma-3-4b-it-4bit_epoch_3 --lora

python src/scripts/model_test.py --model_name lora_finetuned_gemma-3-1b-it-4bit_epoch_4 --lora

python ./src/scripts/model_test.py --model_name finetuned_gemma-3-1b-it-unsloth-bnb-4bit-full
