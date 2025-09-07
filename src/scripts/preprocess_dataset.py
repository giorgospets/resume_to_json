import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

import sys
sys.path.append(os.path.join(PROJECT_ROOT), "src")

import pandas as pd
from utils.text_formatter import TextFormatter


def main():
    INPUT_PATH = os.path.join(PROJECT_ROOT, "data/Resume/Resume.csv")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/preprocessed_dataset.json")

    df = pd.read_csv(INPUT_PATH)
    df["Resume_md"] = df["Resume_html"].apply(TextFormatter.convert_html_to_md)

    df = df[["ID", "Resume_md", "Category"]]
    df.to_json(
        OUTPUT_PATH,
        indent=2,
        force_ascii=False,
        orient="records"
    )


if __name__ == "__main__":
    main()
