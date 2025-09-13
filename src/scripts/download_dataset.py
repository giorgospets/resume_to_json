import requests
import pandas as pd
from pathlib import Path
from io import StringIO


def clean_text(text: str) -> str:
    text = text.replace("\ufeff________________\r\n\r\n", "")
    text = text.replace("\u00ef\u00bb\u00bf________________\r\n\r\n", "")
    text = text.replace("\r\n", "\n")

    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text


def download_and_convert_dataset():
    """Download CSV dataset and convert to JSON format"""
    
    # Dataset URL
    url = "https://github.com/noran-mohamed/Resume-Classification-Dataset/raw/refs/heads/main/Dataset.csv"
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Output file path
    output_file = data_dir / "Dataset.json"
    
    try:
        print("Downloading dataset")
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        
        df["Text"] = df["Text"].apply(clean_text)

        df['ID'] = df.index

        # Reorder columns to put ID first
        cols = ['ID'] + [col for col in df.columns if col != 'ID']
        df = df[cols]
        
        # Convert to JSON
        json_data = df.to_json(orient='records', indent=2)
        
        # Save JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        print(f"Dataset successfully saved to: {output_file}")
        print(f"Dataset contains {len(df)} records with columns: {list(df.columns)}")

        df = pd.read_json(output_file)
        print(df.head())
        
    except requests.RequestException as e:
        print(f"Error downloading dataset: {e}")
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    download_and_convert_dataset()