#!/bin/bash

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

cd ${root_dir}

python ./src/scripts/model_test.py --model_name finetuned_gemma-3-1b-it-unsloth-bnb-4bit-lora-16bit

python ./src/scripts/model_test.py --model_name finetuned_gemma-3-1b-it-unsloth-bnb-4bit-full
