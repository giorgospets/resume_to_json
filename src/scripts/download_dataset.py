import requests
import pandas as pd
import json
import os
import hashlib
from pathlib import Path

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
        print("Downloading dataset...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save CSV temporarily to read with pandas
        temp_csv = data_dir / "temp_dataset.csv"
        with open(temp_csv, 'wb') as f:
            f.write(response.content)
        
        print("Converting CSV to JSON...")
        # Read CSV with pandas
        df = pd.read_csv(temp_csv)
        
        # Add ID field using hash of the row content
        def generate_id(row):
            # Create a string representation of the row
            row_str = ''.join(str(value) for value in row.values)
            # Generate SHA256 hash and take first 8 characters
            return str(hashlib.md5(row_str.encode()).hexdigest())
        
        df['ID'] = df.apply(generate_id, axis=1)
        
        # Reorder columns to put ID first
        cols = ['ID'] + [col for col in df.columns if col != 'ID']
        df = df[cols]
        
        # Convert to JSON
        json_data = df.to_json(orient='records', indent=2)
        
        # Save JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        # Clean up temporary CSV file
        temp_csv.unlink()
        
        print(f"Dataset successfully saved to: {output_file}")
        print(f"Dataset contains {len(df)} records with columns: {list(df.columns)}")
        
    except requests.RequestException as e:
        print(f"Error downloading dataset: {e}")
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    download_and_convert_dataset()