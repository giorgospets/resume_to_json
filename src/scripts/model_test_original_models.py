import json
import os
import sys
import torch
from dotenv import load_dotenv
load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import Levenshtein

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src/scripts"))
sys.path.append(os.path.join(PROJECT_ROOT, "src/utils"))
from utils.training_utils import MAX_SEQ_LENGTH
from utils.training_utils import format_prompts


MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

with open(os.path.join(PROJECT_ROOT, "resume_json_schema.json"), "r") as f:
    json_schema = json.load(f)

SYSTEM_PROMPT = f"""You are an expert in extracting information from CVs and responding with a JSON using the following JSON schema: {json.dumps(json_schema)}"""


def is_json_valid(json_str: str) -> bool:
    try:
        if json_str.startswith("```json\n") and json_str.endswith("```"):
            json_str = json_str[len("```json\n"):-3].strip()
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


def levenshtein_distance(gt: str, pred: str):
    # Levenshtein distance
    dist = Levenshtein.distance(gt, pred)
    
    # Number of matching characters
    matches = max(len(gt), len(pred)) - dist
    
    # Precision & Recall
    precision = matches / len(pred) if pred else 0
    recall = matches / len(gt) if gt else 0
    
    # F1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        "levenshtein_distance": dist,
        "levenshtein_precision": precision,
        "levenshtein_recall": recall,
        "levenshtein_f1": f1
    }


def main():
    """Main function to load data, run inference, and save results."""

    parser = ArgumentParser()

    parser.add_argument("--model_name", default="unsloth/gemma-3-270m-it", type=str)
    args = parser.parse_args()

    # NOTE: Models are:
    # unsloth/gemma-3-270m-it
    # unsloth/gemma-3-1b-it-unsloth-bnb-4bit
    # unsloth/gemma-3-4b-it-unsloth-bnb-4bit

    DATASET_DICT_FILEPATH = os.path.join(PROJECT_ROOT, "data/test_structured_dataset.json")
    OUTPUT_JSON_FILEPATH = os.path.join(PROJECT_ROOT, f"data/test_results_{args.model_name.replace('/', '_')}.json")

    result_dict = {}

    with open(DATASET_DICT_FILEPATH, "r") as f:
        dataset_dict: dict[str, dict] = json.load(f)

    try:
        llm = LLM(
            model=args.model_name,
            dtype="auto",
            max_seq_len_to_capture=MAX_SEQ_LENGTH,
            max_model_len=MAX_SEQ_LENGTH
        )
    except Exception as e:
        print(f"Failed to load model with CUDA: {e}")
        print("Trying CPU-only mode...")
        # Force CPU mode
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        llm = LLM(
            model=args.model_name,
            dtype="auto",
            max_seq_len_to_capture=MAX_SEQ_LENGTH,
            max_model_len=MAX_SEQ_LENGTH,
            device="cpu"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=MAX_SEQ_LENGTH,
        top_k=64
    )

    batch_prompts = []

    for idx, data_point in enumerate(dataset_dict):
        result_dict[idx] = {
            'ID': data_point["ID"],
            "Category": data_point["Category"],
            "Text": data_point["Text"],
            "json": data_point["json"],
            "pred_json": "",
        }

        formatted_prompt = format_prompts(
            system_prompt=SYSTEM_PROMPT,
            cv_input=data_point["Text"],
            tokenizer=tokenizer,
            training_bool=False
        )
        batch_prompts.append(formatted_prompt)

    outputs = llm.generate(batch_prompts, sampling_params)

    for i, output in enumerate(outputs):
        generated_json_str = output.outputs[0].text.strip()
        result_dict[i]["pred_json"] = generated_json_str
        result_dict[i].update(levenshtein_distance(
            result_dict[i]["json"], 
            generated_json_str
        ))
        result_dict[i]["is_json_valid"] = is_json_valid(generated_json_str)

    print(f"Saving results to {OUTPUT_JSON_FILEPATH}")
    with open(OUTPUT_JSON_FILEPATH, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    print("Processing complete. Results have been saved.")


if __name__ == "__main__":
    main()
