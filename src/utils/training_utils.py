from typing import Any, List, Dict, Optional
import os
from datasets import Dataset
import json
from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
TRAIN_DATASET_DICT_PATH = os.path.join(PROJECT_ROOT, "data/train_structured_dataset.json")
VAL_DATASET_DICT_PATH = os.path.join(PROJECT_ROOT, "data/val_structured_dataset.json")

MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_PATH, exist_ok=True)


with open(os.path.join(PROJECT_ROOT, "resume_json_schema.json"), "r") as f:
    JSON_SCHEMA = json.load(f)


MAX_SEQ_LENGTH = 8192



def format_prompts(
    system_prompt: str,
    cv_input: str,
    tokenizer: Any,
    ground_truth_json: Optional[str] = None,
    training_bool: bool = False
) -> str:
    """
    Formats the input CV and ground truth JSON into a
    structured conversation format using the tokenizer's chat template.
    """
    # Construct the message list for the chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": cv_input},
    ]

    # If creating a training example, add the assistant's correct response
    if training_bool:
        messages.append({"role": "assistant", "content": ground_truth_json})

    formatted_conv = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not training_bool,
    )
    return formatted_conv

def create_resume_dataset(
    data: List[Dict[str, Any]], 
    tokenizer: Any,
    system_prompt: str,
    training_bool: bool = True
) -> Dataset:
    """
    Prepares a resume dataset for fine-tuning by applying the chat template
    via the `format_prompts` function.
    """
    def format_examples(
        examples: Dict[str, List],
        training_bool: bool = True
    ) -> Dict[str, List[str]]:
        """Applies the prompt formatting to a batch of examples."""
        texts = []
        for i in range(len(examples["Text"])):
            cv_text = examples["Text"][i]
            # Pass the ground-truth dictionary directly
            json_output = examples["json"][i]

            formatted_text = format_prompts(
                system_prompt=system_prompt,
                cv_input=cv_text,
                ground_truth_json=json_output,
                tokenizer=tokenizer,
                training_bool=training_bool,
            )
            texts.append(formatted_text)

        return {"text": texts}
    
    def filter_long_sequences(example):
        try:
            return len(tokenizer.encode(example['text'])) <= MAX_SEQ_LENGTH
        except AttributeError:
            return len(tokenizer(example['text'])) <= MAX_SEQ_LENGTH

    dataset = Dataset.from_list(data)

    formatted_dataset = dataset.map(
        format_examples,
        batched=True,
        fn_kwargs={"training_bool": training_bool},
        remove_columns=dataset.column_names,
    )
    return formatted_dataset.filter(filter_long_sequences)
