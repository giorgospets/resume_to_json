# from typing import Any, List, Dict, Optional
# from unsloth import FastModel
# import torch
# import os
# from datasets import Dataset
# import json
# from dotenv import load_dotenv
# from trl import SFTTrainer
# from transformers import TrainingArguments
# from unsloth import is_bfloat16_supported
# from unsloth.chat_templates import train_on_responses_only
# from unsloth.chat_templates import get_chat_template

# max_seq_length = 9000

# load_dotenv()

# PROJECT_ROOT = os.getenv("PROJECT_ROOT")
# DATASET_DICT_PATH = os.path.join(PROJECT_ROOT, "data/train_structured_dataset.json")

# MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
# os.makedirs(MODELS_PATH, exist_ok=True)

# print(DATASET_DICT_PATH)
# with open(os.path.join(PROJECT_ROOT, "resume_json_schema.json"), "r") as f:
#     json_schema = json.load(f)

# SYSTEM_PROMPT = f"""You are an expert in extracting information from CVs and responding with a JSON using the following JSON schema: {json.dumps(json_schema)}


# ##### RULES:
# - Always respond with a valid JSON; do not provide any extra information.
# - Information should be **extracted** from the CV: All characters should remain the same, even whitespace characters. Paraphrasing, implying are not allowed.
# - Only fill values if the information can be found within the CV. 
# - Missing information should be noted either as "" or [], depending on the JSON schema.
# - Personal urls can be urls for the person's linkdenin, github, personal page etc. Urls should be in valid url format; e.g. The following is a valid personal_url: linkedin.com/in/cristiano-ronaldo/. The following are *not* valid personal_url's: LinkedIn, Twitter, in LinkedIn (because they contain no personal information).
# """

# USER_PROMPT = """##### CV:
# <cv>{}</cv>
# """


# def format_prompts(
#     cv_input: str,
#     tokenizer: Any,
#     ground_truth_json: Optional[str] = None,
#     training_bool: bool = False
# ) -> str:
#     """
#     Formats the input CV and ground truth JSON into a
#     structured conversation format using the tokenizer's chat template.
#     """
#     # Construct the message list for the chat template
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": USER_PROMPT.format(cv_input)},
#     ]

#     # If creating a training example, add the assistant's correct response
#     if training_bool:
#         messages.append({"role": "assistant", "content": ground_truth_json})

#     # Apply chat template without tokenizing first
#     formatted_conv = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=not training_bool,
#     )
#     return formatted_conv


# # 3. The main function to create the dataset
# # -------------------------------------------------
# def create_resume_dataset(data: List[Dict[str, Any]], tokenizer: Any) -> Dataset:
#     """
#     Prepares a resume dataset for fine-tuning by applying the chat template
#     via the `format_prompts` function.
#     """
#     dataset = Dataset.from_list(data)

#     def format_examples(examples: Dict[str, List]) -> Dict[str, List[str]]:
#         """Applies the prompt formatting to a batch of examples."""
#         texts = []
#         for i in range(len(examples["Text"])):
#             cv_text = examples["Text"][i]
#             # Pass the ground-truth dictionary directly
#             json_output = examples["json"][i]

#             formatted_text = format_prompts(
#                 cv_input=cv_text,
#                 ground_truth_json=json_output,
#                 tokenizer=tokenizer,
#                 training_bool=True
#             )
#             texts.append(formatted_text)

#         return {"text": texts}

#     # Apply the mapping function to format every example in the dataset
#     processed_dataset = dataset.map(
#         format_examples,
#         batched=True,
#         remove_columns=dataset.column_names,  # Keep only the final 'text' column
#     )

#     return processed_dataset


# model, tokenizer = FastModel.from_pretrained(
#     model_name="unsloth/gemma-3-270m-it",
#     max_seq_length=max_seq_length,
#     load_in_4bit=False,
#     load_in_8bit=False,
#     full_finetuning=True,
#     dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
# )


# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="gemma-3",
# )


# with open(DATASET_DICT_PATH, "r") as f:
#     dataset_dict = json.load(f)

# dataset = create_resume_dataset(dataset_dict, tokenizer)

# max_seq_length = 8192


# def filter_long_sequences(example, max_seq_length):
#     return len(tokenizer.encode(example['text'])) <= max_seq_length


# dataset = dataset.filter(filter_long_sequences, max_seq_length)

# output_dir = os.path.join(MODELS_PATH, "full_finetuned_gemma-3-270m-it")
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=max_seq_length,
#     dataset_num_proc=2,
#     packing=True,
#     args=TrainingArguments(
#         per_device_train_batch_size=8,
#         gradient_accumulation_steps=16,
#         warmup_ratio=0.05,
#         num_train_epochs=5,
#         learning_rate=2e-5,
#         fp16=not is_bfloat16_supported(),
#         bf16=is_bfloat16_supported(),
#         # fp16=True,
#         logging_steps=1,
#         optim="adamw_torch",
#         weight_decay=0.01,
#         lr_scheduler_type="linear",
#         seed=3407,
#         output_dir=output_dir,
#         report_to="none",
#         save_strategy="epoch",
#         save_total_limit=1,
#     ),
# )

# trainer = train_on_responses_only(
#     trainer,
#     instruction_part="<start_of_turn>user\n",
#     response_part="<start_of_turn>model\n",
# )

# # GPU stats and training call
# gpu_stats = torch.cuda.get_device_properties(0)
# start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
# print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
# print(f"{start_gpu_memory} GB of memory reserved.")

# trainer_stats = trainer.train()



# finetuned_model_dir = os.path.join(MODELS_PATH,"finetuned_gemma-3-1b-it-unsloth-bnb-4bit-full")
# model.save_pretrained(finetuned_model_dir)
# tokenizer.save_pretrained(finetuned_model_dir)

# # Also save the training arguments and config
# model.config.to_json_file(os.path.join(finetuned_model_dir, "config.json"))
