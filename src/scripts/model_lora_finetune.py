from unsloth import FastModel
import torch
import os
import json
from trl import SFTTrainer
from transformers import TrainingArguments
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


load_in_4bit = True

SYSTEM_PROMPT = f"""You are an expert in extracting information from CVs and responding with a JSON using the following JSON schema: {json.dumps(JSON_SCHEMA)}


##### RULES:
- Always respond with a valid JSON; do not provide any extra information.
- Information should be **extracted** from the CV: All characters should remain the same, even whitespace characters. Paraphrasing, implying are not allowed.
- Only fill values if the information can be found within the CV. 
- Missing information should be noted either as "" or [], depending on the JSON schema.
"""


model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
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
    args = TrainingArguments(
        per_device_train_batch_size=12,
        gradient_accumulation_steps=24,
        warmup_ratio=0.05,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=os.path.join(MODELS_PATH, "lora_finetuned_gemma-3-1b-it-4bit"),
        report_to="none",
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=True,
        args=args
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # GPU stats and training call
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()
