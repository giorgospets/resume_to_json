import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

import sys
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import pandas as pd
from  utils.dataset_utils import clean_text, add_id_column


if __name__ == "__main__":
    df = pd.read_json(os.path.join(PROJECT_ROOT, "data/Dataset.json"))
    df = add_id_column(df)
    df["Text"] = df["Text"].apply(clean_text)
    df.to_json(os.path.join(PROJECT_ROOT, "data/Dataset.json"), orient='records', indent=2)

