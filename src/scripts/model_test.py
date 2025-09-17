import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
from typing import Any, Optional
import os
from dotenv import load_dotenv
from utils.training_utils import MAX_SEQ_LENGTH

load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
with open(os.path.join(PROJECT_ROOT, "resume_json_schema.json"), "r") as f:
    json_schema = json.load(f)

SYSTEM_PROMPT = f"""You are an expert in extracting information from CVs and responding with a JSON using the following JSON schema: {json.dumps(json_schema)}

##### RULES:
- Always respond with a valid JSON; do not provide any extra information.
- Information should be **extracted** from the CV: All characters should remain the same, even whitespace characters. Paraphrasing, implying are not allowed.
- Only fill values if the information can be found within the CV. 
- Missing information should be noted either as "" or [], depending on the JSON schema.
- Personal urls can be urls for the person's linkdenin, github, personal page etc. Urls should be in valid url format; e.g. The following is a valid personal_url: linkedin.com/in/cristiano-ronaldo/. The following are *not* valid personal_url's: LinkedIn, Twitter, in LinkedIn (because they contain no personal information).
"""

USER_PROMPT = """##### CV:
<cv>{}</cv>
"""


def format_prompts(
        cv_input: str,
        tokenizer: Any,
        ground_truth_json: Optional[str] = None,
        training_bool: bool = False
) -> str:
    """
    Formats the input CV and ground truth JSON into a
    structured conversation format using the tokenizer's chat template.
    """
    # Create a string from the list of documents, or "None" if it's empty.

    # Construct the message list for the chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(cv_input)},
    ]

    # If creating a training example, add the assistant's correct response
    if training_bool:
        # Use a JSON markdown block for the ground truth, which is standard practice.
        assistant_response = f"```json\n{ground_truth_json}\n```"
        messages.append({"role": "assistant", "content": assistant_response})

    # The tokenizer's template handles the specific formatting for the model
    # (e.g., adding special tokens like <|start_header_id|>).
    formatted_conv = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not training_bool,
    )
    return formatted_conv


def main():
    """Main function to load data, run inference, and save results."""

    parser = ArgumentParser()

    parser.add_argument("--model_name", default="finetuned_gemma-3-1b-it-unsloth-bnb-4bit-lora-16bit", type=str)
    args = parser.parse_args()


    DATASET_DICT_FILEPATH = os.path.join(PROJECT_ROOT, "data/test_structured_dataset.json")
    OUTPUT_JSON_FILEPATH = os.path.join(PROJECT_ROOT, f"data/test_results_{args.model_name.split("/")[1].split("/")[0]}.json")

    result_dict = {}

    with open(DATASET_DICT_FILEPATH, "r") as f:
        dataset_dict: dict[str, dict] = json.load(f)

    finetuned_lora_model_dir_16_bit = os.path.join(MODELS_PATH, args.model_name)

    llm = LLM(
        model=finetuned_lora_model_dir_16_bit,
        dtype="auto",
        max_seq_len_to_capture=MAX_SEQ_LENGTH
    )
    tokenizer = AutoTokenizer.from_pretrained(finetuned_lora_model_dir_16_bit)


    sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=150,
            top_k=64
        )

    batch_prompts = []

    for idx, data_point in enumerate(dataset_dict):
        result_dict[idx] = {
            'ID': data_point["ID"],
            "Category": data_point["Category"],
            "Text": data_point["Text"],
            "json": data_point["json"],
            "pred_json": ""
        }

        formatted_prompt = format_prompts(
            cv_input=data_point["Text"],
            tokenizer=tokenizer,
        )
        batch_prompts.append(formatted_prompt)

    outputs = llm.generate(batch_prompts, sampling_params)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        result_dict[i]["pred_json"] = generated_text

    print(f"Saving results to {OUTPUT_JSON_FILEPATH}")
    with open(OUTPUT_JSON_FILEPATH, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    print("Processing complete. Results have been saved.")


if __name__ == "__main__":
    main()
