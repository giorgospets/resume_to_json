"""Post-process the created dataset to ensure all JSON objects conform to the schema."""

import json
import os
from dotenv import load_dotenv

load_dotenv()


def fill_json_schema(schema, data):
    if not isinstance(data, dict):
        print(f"Error: Expected dictionary for recursive call, but received '{type(data).__name__}'.")
        return {}

    filled_data = data.copy()

    for key in list(filled_data.keys()):
        if key not in schema:
            print(f"Warning: Key '{key}' in the input data is not present in the schema. Removing it from the output.")
            del filled_data[key]

    for key, schema_value in schema.items():
        if key not in filled_data:
            print(f"Warning: Key '{key}' in the schema is not present in the input data. Adding it to the output.")
            if isinstance(schema_value, str) and schema_value == "string":
                filled_data[key] = ""
            elif isinstance(schema_value, dict):
                filled_data[key] = {}
            elif isinstance(schema_value, list):
                filled_data[key] = []
        else:
            # If the key exists, but the value is a dictionary or list,
            # recursively call the function to check for nested missing keys.
            if isinstance(schema_value, dict) and isinstance(filled_data[key], dict):
                filled_data[key] = fill_json_schema(schema_value, filled_data[key])
            elif isinstance(schema_value, list) and isinstance(filled_data[key], list):
                # The schema for lists is a template. We don't want to create
                # a new item, just ensure the existing ones are complete.
                if len(schema_value) > 0 and isinstance(schema_value[0], dict):
                    item_schema = schema_value[0]
                    for i in range(len(filled_data[key])):
                        # Only recursively call if the item is a dictionary
                        if isinstance(filled_data[key][i], dict):
                            filled_data[key][i] = fill_json_schema(item_schema, filled_data[key][i])
                        else:
                            print(f"Warning: Item at index {i} in list '{key}' is not a dictionary as expected by the schema.")
                            filled_data[key][i] = {}
    return filled_data

def clean_non_url_string(json: dict) -> dict:
    """Remove non-URL strings from the personal_urls field in personal_information."""
    def is_url_valid(url: str) -> bool:
        if len(url.split()) > 1:
            return False
        for st in [".com", "http", "www"]:
            if st in url:
                return True
        return False
        
    valid_urls = [] 
    for url in json.get("personal_information", {}).get("personal_urls", []):
        if is_url_valid(url):
            valid_urls.append(url)

    json["personal_information"]["personal_urls"] = valid_urls
    return json

def remove_empty_json(dataset: list[dict]) -> dict:
    """Remove entries with empty json"""
    indices_to_remove = []
    for idx, datapoint in enumerate(dataset):
        if not datapoint['json']:
            indices_to_remove.append(idx)

    return [val for idx, val in enumerate(dataset) if idx not in indices_to_remove]

if __name__ == "__main__":
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    DATASET_DICT_FILEPATH = os.path.join(PROJECT_ROOT, "data/orig_structured_dataset.json")
    MODIFIED_DATASET_DICT_FILEPATH = os.path.join(PROJECT_ROOT, "data/structured_dataset.json")

    with open(os.path.join(PROJECT_ROOT, "resume_json_schema.json"), "r") as f:
        json_schema: dict = json.load(f)

    with open(DATASET_DICT_FILEPATH, "r") as f:
        dataset: list[dict] = json.load(f)

    dataset = remove_empty_json(dataset)

    for datapoint in dataset:
        datapoint['json'] = fill_json_schema(json_schema, datapoint['json'])
        datapoint['json'] = json.dumps(datapoint['json'])
        
    with open(MODIFIED_DATASET_DICT_FILEPATH, "w") as f:
        json.dump(dataset, f)

    print("finished")
