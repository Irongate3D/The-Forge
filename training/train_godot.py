"""
Irongate — Godot AI Specialist
Unsloth LoRA fine-tune: Qwen2.5-3B-Instruct on godot_final.jsonl
Run from WSL2 with unsloth-env active:
    python train_godot.py
"""

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# ─── PATHS ────────────────────────────────────────────────────────────────────
# Defaults match the author's WSL2 layout; override with environment variables.
MODEL_PATH   = os.environ.get("GODOT_BASE_MODEL", "/mnt/d/Irongate/models/Qwen2.5-3B-Instruct")
DATASET_PATH = os.environ.get("GODOT_DATASET", "/mnt/d/Irongate/Forge/domains/godot/datasets/godot_final.jsonl")
OUTPUT_PATH  = os.environ.get("GODOT_LORA_OUT", "/mnt/d/Irongate/models/godot-lora")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 2048   # Qwen2.5 supports up to 32k but 2048 is safe for first run
LOAD_IN_4BIT   = True   # Keeps VRAM usage low (~2-3GB for 3B)
LORA_RANK      = 16     # Standard starting point; increase to 32 for more capacity

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_PATH,
    max_seq_length  = MAX_SEQ_LENGTH,
    dtype           = None,   # Auto-detect: float16 for ROCm
    load_in_4bit    = LOAD_IN_4BIT,
    local_files_only= True,   # Don't phone home to HuggingFace
)

# ─── APPLY LORA ───────────────────────────────────────────────────────────────
print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r                   = LORA_RANK,
    target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    lora_alpha          = 16,
    lora_dropout        = 0,    # 0 is optimised in Unsloth
    bias                = "none",
    use_gradient_checkpointing = "unsloth",  # Saves VRAM
    random_state        = 42,
)

# ─── DATASET ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"  {len(dataset)} training pairs loaded")

# Alpaca-style prompt template for Qwen2.5
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def format_prompts(examples):
    instructions = examples["instruction"]
    # The Forge writes "response"; the training set used "output". Accept either.
    outputs      = examples["output"] if "output" in examples else examples["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = ALPACA_TEMPLATE.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

# ─── TRAINER ──────────────────────────────────────────────────────────────────
print("Setting up trainer...")
trainer = SFTTrainer(
    model        = model,
    tokenizer    = tokenizer,
    train_dataset= dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    dataset_num_proc   = 2,
    args = SFTConfig(
        per_device_train_batch_size   = 2,
        gradient_accumulation_steps   = 4,   # Effective batch = 8
        warmup_steps                  = 50,
        num_train_epochs              = 3,
        learning_rate                 = 2e-4,
        fp16                          = not torch.cuda.is_bf16_supported(),
        bf16                          = torch.cuda.is_bf16_supported(),
        logging_steps                 = 25,
        optim                         = "adamw_8bit",
        weight_decay                  = 0.01,
        lr_scheduler_type             = "linear",
        seed                          = 42,
        output_dir                    = OUTPUT_PATH,
        save_strategy                 = "epoch",
        save_total_limit              = 1,
        report_to                     = "none",  # No wandb
    ),
)

# ─── TRAIN ────────────────────────────────────────────────────────────────────
print("Starting training...")
print(f"  Dataset: {len(dataset)} pairs")
print(f"  Epochs:  3")
print(f"  Output:  {OUTPUT_PATH}")
print()

trainer_stats = trainer.train()

print()
print(f"Training complete in {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"Peak VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.1f} GB")

# ─── SAVE ─────────────────────────────────────────────────────────────────────
print("Saving LoRA adapters...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"Adapters saved to {OUTPUT_PATH}")
print()
print("Next step: run merge_to_gguf.py to merge adapters and export for LM Studio")