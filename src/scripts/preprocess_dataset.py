import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

import sys
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import pandas as pd
from  utils.dataset_utils import clean_text, add_id_column, replace_job_title, replace_language, remove_skill


if __name__ == "__main__":
    df = pd.read_json(os.path.join(PROJECT_ROOT, "data/dataset.json"))
    df = add_id_column(df)
    df["Text"] = df["Text"].apply(clean_text)
    replace_job_title(df)
    replace_language(df)
    df = remove_skill(df)
    df.to_json(os.path.join(PROJECT_ROOT, "data/preprocessed_dataset.json"), orient='records', indent=2)

