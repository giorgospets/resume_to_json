#!/bin/bash

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

cd ${root_dir}

python ./src/scripts/download_dataset.py

python ./src/scripts/preprocess_dataset.py

python ./src/scripts/create_dataset.py

python ./src/scripts/postprocess_created_dataset.py

python ./src/scripts/split_dataset.py