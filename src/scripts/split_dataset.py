
"""Split the structured dataset into training and testing sets and save them as separate JSON files."""

import json
import os
import os
import json
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    DATASET_FILEPATH = os.path.join(PROJECT_ROOT, "data/structured_dataset.json")
    PARENT_DIR = os.path.dirname(DATASET_FILEPATH)

    with open(DATASET_FILEPATH, "r") as f:
        dataset = json.load(f)

    train_val_data, test_data = train_test_split(
        dataset, 
        test_size=100, 
        random_state=42
    )

    train_data, val_data = train_test_split(
        train_val_data, 
        test_size=100, 
        random_state=42
    )

    train_filepath = os.path.join(PARENT_DIR, "train_structured_dataset.json")
    test_filepath = os.path.join(PARENT_DIR, "test_structured_dataset.json")
    val_filepath = os.path.join(PARENT_DIR, "val_structured_dataset.json")

    with open(train_filepath, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(val_filepath, "w") as f:
        json.dump(val_data, f, indent=2)

    with open(test_filepath, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Dataset split completed!")
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Train file saved to: {train_filepath}")
    print(f"Test file saved to: {test_filepath}")
    print(f"Val set saved to: {val_filepath}")
