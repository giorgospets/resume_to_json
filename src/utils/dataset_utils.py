import pandas as pd
import re
import random
import json
from pathlib import Path
import os
import numpy as np

def add_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df['ID'] = range(1, len(df) + 1)

    # Reorder columns to put ID first
    cols = ['ID'] + [col for col in df.columns if col != 'ID']
    df = df[cols]

    return df


def clean_text(text: str) -> str:
    text = text.replace("\ufeff________________\r\n\r\n", "")
    text = text.replace("\u00ef\u00bb\u00bf________________\r\n\r\n", "")
    text = text.replace("\r\n", "\n")

    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text


def find_discrete_anonymized_brackets(df: pd.DataFrame) -> dict:
    """Returns a dict of texts that are in the form of [text] in the Text column amd their counts
    
    Example of Text: 
    ...SUMMARY\nExperienced [Job Title] with over [Number] years of experience in [Industry]...
    """
    pattern = r'[([^\]]+)]'
    all_matches = []
    for text in df['Text']:
        matches = re.findall(pattern, text)
        matches = [x.strip().lower() for x in matches]
        all_matches.extend(matches)

    all_matches.sort()

    match_counts = {match: all_matches.count(match) for match in all_matches}
    match_counts = {match: count for match, count in match_counts.items() if count >= 10}
    return match_counts


def replace_job_title(df: pd.DataFrame) -> None:
    with open(os.path.join(Path(__file__).parent, 'category_to_job_titles_dict.json')) as f:
        category_to_job_titles_dict = json.load(f)

    job_title_comb = ['[job title]', '[Job Title]', '[Job title]', '[job Title]', '[JOB TITLE]']
    for _, row in df.iterrows():
        if not any([jb_title_cb in row['Text'] for jb_title_cb in job_title_comb]):
            continue
        for jb_title_cb in job_title_comb:
            if jb_title_cb in row['Text']:
                random_job_title = random.choice(category_to_job_titles_dict[row['Category']])
                row['Text'] = row['Text'].replace(jb_title_cb, random_job_title)
    return None

def remove_skill(df: pd.DataFrame) -> None:
    skills_comb = ['[Skill]', '[skill]', '[SKILL]']
    not_found_skill = [not any([sk_cb in row['Text'] for sk_cb in skills_comb]) for _, row in df.iterrows()]
    return df[not_found_skill]

def replace_language(df: pd.DataFrame) -> None:
    languages = [
        "English", "Spanish", "Mandarin Chinese", "Hindi", "Arabic", "Bengali", "Portuguese", "Russian", "Japanese",
        "Punjabi", "German", "Javanese", "French", "Turkish", "Korean", "Greek"
    ]

    language_comb = ['[language]', '[Language]', '[LANGUAGE]']

    rows_found = 0
    for _, row in df.iterrows():
        if not any([lang_cb in row['Text'] for lang_cb in language_comb]):
            continue
        rows_found += 1
        for lang_cb in language_comb:
            if lang_cb in row['Text']:
                occurrences = row['Text'].count(lang_cb)
                idx_languages_found = []
                for idx, language in enumerate(languages):
                    if language.lower() in row['Text'].lower():
                        idx_languages_found.append(idx)
                curr_languages = [v for i, v in enumerate(languages) if i not in idx_languages_found]

                random_languages = np.random.choice(curr_languages, replace=False, size=occurrences)
                for occurrence in range(occurrences):
                    row['Text'] = row['Text'].replace(lang_cb, random_languages[occurrence])
    return None

if __name__ == "__main__":
    # Example usage
    df = pd.read_json("data/Dataset.json")
    df = add_id_column(df)
    df["Text"] = df["Text"].apply(clean_text)
    with open("data/discrete_anonymized_brackets.txt", "w") as f:
        for item, count in find_discrete_anonymized_brackets(df).items():
            f.write(f"{item}: {count}\n")
