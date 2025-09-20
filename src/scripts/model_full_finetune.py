from unsloth import FastModel
import torch
import os
import json
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import get_chat_template

from utils.training_utils import (
    create_resume_dataset,
    MAX_SEQ_LENGTH,
    TRAIN_DATASET_DICT_PATH,
    VAL_DATASET_DICT_PATH,
    MODELS_PATH,
    JSON_SCHEMA,
)

SYSTEM_PROMPT = f"""You are an expert in extracting information from CVs and responding with a JSON using the following JSON schema: {json.dumps(JSON_SCHEMA)}"""


model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=True,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
)


tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)


with open(TRAIN_DATASET_DICT_PATH, "r") as f:
    train_dataset = json.load(f)
with open(VAL_DATASET_DICT_PATH, "r") as f:
    val_dataset = json.load(f)

train_dataset = create_resume_dataset(train_dataset, tokenizer, SYSTEM_PROMPT)
val_dataset = create_resume_dataset(val_dataset, tokenizer, SYSTEM_PROMPT)


if __name__ == "__main__":
    args = TrainingArguments(
        per_device_train_batch_size=12,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=24,
        warmup_ratio=0.05,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=os.path.join(MODELS_PATH, "full_finetuned_gemma-3-270m-it"),
        report_to="tensorboard",
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        max_grad_norm=1.0,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
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
