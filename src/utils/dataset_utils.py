import pandas as pd
import re


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
    pattern = r'\[([^\]]+)\]'
    all_matches = []
    for text in df['Text']:
        matches = re.findall(pattern, text)
        matches = [x.strip().lower() for x in matches]
        all_matches.extend(matches)

    all_matches.sort()

    # Count occurrences of each match
    match_counts = {match: all_matches.count(match) for match in all_matches}
    match_counts = {match: count for match, count in match_counts.items() if count >= 10}
    return match_counts

if __name__ == "__main__":
    # Example usage
    df = pd.read_json("data/Dataset.json")
    df = add_id_column(df)
    df["Text"] = df["Text"].apply(clean_text)
    with open("data/discrete_anonymized_brackets.txt", "w") as f:
        for item, count in find_discrete_anonymized_brackets(df).items():
            f.write(f"{item}: {count}\n")
