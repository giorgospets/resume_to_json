#!/bin/bash

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"

mkdir -p ${root_dir}/data/

cd ${root_dir}/data

curl -L -o resume-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/snehaanbhawal/resume-dataset

unzip resume-dataset.zip

rm resume-dataset.zip