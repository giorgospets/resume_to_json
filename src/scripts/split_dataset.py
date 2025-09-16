import json
from dotenv import load_dotenv
import os
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
import os
import json
import random

# Set seed for reproducibility
random.seed(42)

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
DATASET_DICT_FILEPATH = os.path.join(PROJECT_ROOT, "data/structured_dataset.json")

# Load the dataset
with open(DATASET_DICT_FILEPATH, "r") as f:
    dataset_dict = json.load(f)

# Process the data as in your original code
for datapoint in dataset_dict:
    datapoint['json'] = json.dumps(datapoint['json'])

# Shuffle the dataset
random.shuffle(dataset_dict)

# Split into train and test (150 samples for test)
test_size = 150
train_data = dataset_dict[:-test_size]
test_data = dataset_dict[-test_size:]

# Get the parent directory of the original file
parent_dir = os.path.dirname(DATASET_DICT_FILEPATH)

# Save train and test files
train_filepath = os.path.join(parent_dir, "train_structured_dataset.json")
test_filepath = os.path.join(parent_dir, "test_structured_dataset.json")

with open(train_filepath, "w") as f:
    json.dump(train_data, f, indent=2)

with open(test_filepath, "w") as f:
    json.dump(test_data, f, indent=2)

print(f"Dataset split completed!")
print(f"Total samples: {len(dataset_dict)}")
print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Train file saved to: {train_filepath}")
print(f"Test file saved to: {test_filepath}")