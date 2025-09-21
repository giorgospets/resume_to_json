from unsloth import FastModel
import torch
import os
import sys
from dotenv import load_dotenv
import json
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from argparse import ArgumentParser
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src/scripts"))
sys.path.append(os.path.join(PROJECT_ROOT, "src/utils"))

from utils.training_utils import (
    create_resume_dataset,
    MAX_SEQ_LENGTH,
    TRAIN_DATASET_DICT_PATH,
    VAL_DATASET_DICT_PATH,
    MODELS_PATH,
    JSON_SCHEMA,
)


load_in_4bit = True

SYSTEM_PROMPT = f"""You are an expert in extracting information from CVs and responding with a JSON using the following JSON schema: {json.dumps(JSON_SCHEMA)}"""


model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
)

# Attach LoRA adapters
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=64,
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)


tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)


with open(TRAIN_DATASET_DICT_PATH, "r") as f:
    train_dataset_dict = json.load(f)
with open(VAL_DATASET_DICT_PATH, "r") as f:
    val_dataset_dict = json.load(f)

train_dataset = create_resume_dataset(train_dataset_dict, tokenizer, SYSTEM_PROMPT)
val_dataset = create_resume_dataset(val_dataset_dict, tokenizer, SYSTEM_PROMPT)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", default=1, type=float)
    parser_args = parser.parse_args()
    model_name = 'lora_finetuned_gemma-3-4b-it-4bit'

    args = TrainingArguments(
        per_device_train_batch_size=12,
        gradient_accumulation_steps=24,
        warmup_ratio=0.05,
        num_train_epochs=parser_args.num_epochs,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=os.path.join(MODELS_PATH, model_name),
        report_to="tensorboard",
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
    )


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=12,
        packing=True,
        args=args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.005
            )
        ],
    )

    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part="<start_of_turn>user\n",
    #     response_part="<start_of_turn>model\n",
    # )

    # GPU stats and training call
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    finetuned_lora_model_dir_16_bit = os.path.join(MODELS_PATH, f"{model_name}_epoch_{parser_args.num_epochs}")
    model.save_pretrained_merged(finetuned_lora_model_dir_16_bit, tokenizer, save_method="merged_16bit")