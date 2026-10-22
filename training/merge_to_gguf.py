"""
Irongate — Godot AI Specialist
Merge LoRA adapters into base model, then export to GGUF for LM Studio.

Run from WSL2 with unsloth-env active:
    source ~/unsloth-env/bin/activate
    python merge_to_gguf.py

llama.cpp must be built first (separate llama-env, not unsloth-env).
"""

import os
import sys
import subprocess
from unsloth import FastLanguageModel

# ─── PATHS ────────────────────────────────────────────────────────────────────
# Defaults match the author's WSL2 layout; override with environment variables.
LORA_PATH     = os.environ.get("GODOT_LORA_CHECKPOINT", "/mnt/d/Irongate/models/godot-lora/checkpoint-1910")  # Trained adapters
MERGED_PATH   = os.environ.get("GODOT_MERGED_OUT", "/mnt/d/Irongate/models/godot-merged")                    # Merged full model
GGUF_PATH     = os.environ.get("GODOT_GGUF_OUT", "/mnt/d/Irongate/models/godot-gguf")                        # Final GGUF output
GGUF_NAME     = "godot-specialist-q4.gguf"                                                                   # Output filename
LLAMA_CPP     = os.environ.get("LLAMA_CPP", "/mnt/d/Irongate/llama.cpp")                                     # llama.cpp repo path
LLAMA_PYTHON  = os.environ.get("LLAMA_ENV_PYTHON", "/root/llama-env/bin/python")                             # llama-env Python
LLAMA_PYTHON2 = os.path.expanduser("~/llama-env/bin/python")                                                 # fallback path
KEEP_F16      = os.environ.get("KEEP_F16", "") == "1"   # set KEEP_F16=1 to keep the f16 GGUF after quantising

MAX_SEQ_LENGTH = 2048

# ─── RESOLVE LLAMA PYTHON ─────────────────────────────────────────────────────
convert_script = os.path.join(LLAMA_CPP, "convert_hf_to_gguf.py")
quantize_bin   = os.path.join(LLAMA_CPP, "build", "bin", "llama-quantize")

llama_python = None
for candidate in [LLAMA_PYTHON, LLAMA_PYTHON2]:
    if os.path.exists(candidate):
        llama_python = candidate
        break

if not llama_python:
    print("=" * 60)
    print("llama-env not found. Create it first (in a separate terminal):")
    print()
    print("  python3 -m venv ~/llama-env")
    print("  source ~/llama-env/bin/activate")
    print("  pip install -r /mnt/d/Irongate/llama.cpp/requirements.txt")
    print("  deactivate")
    print("=" * 60)
    sys.exit(1)

if not os.path.exists(convert_script):
    print("=" * 60)
    print("llama.cpp not found or not built at:", LLAMA_CPP)
    print()
    print("  cd /mnt/d/Irongate")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print("  cd llama.cpp")
    print("  cmake -B build -DGGML_HIP=ON \\")
    print("    -DGPU_TARGETS=gfx1100 \\")
    print("    -DCMAKE_BUILD_TYPE=Release")
    print("  cmake --build build --config Release -j$(nproc)")
    print("  (gfx1100 = RX 7900 XTX/XT. See llama.cpp docs/build.md for your GPU and ROCm setup.)")
    print("=" * 60)
    sys.exit(1)

print(f"Using llama-env Python: {llama_python}")
print(f"Using convert script:   {convert_script}")
print(f"Using quantize binary:  {quantize_bin}")

# ─── STEP 1: LOAD BASE + ADAPTERS ─────────────────────────────────────────────
print()
print("=" * 60)
print("Step 1: Loading base model + LoRA adapters...")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = LORA_PATH,
    max_seq_length   = MAX_SEQ_LENGTH,
    dtype            = None,
    load_in_4bit     = True,
    local_files_only = True,
)

# ─── STEP 2: MERGE AND SAVE AS FLOAT16 ────────────────────────────────────────
print()
print("=" * 60)
print("Step 2: Merging adapters into base model (float16)...")
print("This writes ~6GB to disk — may take a few minutes on NTFS mount.")
print("=" * 60)

os.makedirs(MERGED_PATH, exist_ok=True)

model.save_pretrained_merged(
    MERGED_PATH,
    tokenizer,
    save_method = "merged_16bit",
)

print(f"Merged model saved to {MERGED_PATH}")

# ─── STEP 3: CONVERT TO GGUF ──────────────────────────────────────────────────
print()
print("=" * 60)
print("Step 3: Converting to GGUF (f16) using llama-env...")
print("=" * 60)

os.makedirs(GGUF_PATH, exist_ok=True)
gguf_f16 = os.path.join(GGUF_PATH, "godot-specialist-f16.gguf")

convert_cmd = [
    llama_python, convert_script,
    MERGED_PATH,
    "--outfile", gguf_f16,
    "--outtype", "f16",
]

print("Running:", " ".join(convert_cmd))
subprocess.run(convert_cmd, check=True)
print(f"F16 GGUF saved to {gguf_f16}")

# ─── STEP 4: QUANTIZE TO Q4_K_M ──────────────────────────────────────────────
print()
print("=" * 60)
print("Step 4: Quantizing to Q4_K_M...")
print("Q4_K_M = best quality/size balance for LM Studio.")
print("=" * 60)

gguf_q4 = os.path.join(GGUF_PATH, GGUF_NAME)

if not os.path.exists(quantize_bin):
    print()
    print("llama-quantize binary not found — skipping quantization.")
    print("Your F16 GGUF is usable in LM Studio as-is (~6GB).")
    print(f"File: {gguf_f16}")
    print()
    print("To quantize manually later:")
    print(f"  {quantize_bin} {gguf_f16} {gguf_q4} Q4_K_M")
else:
    quant_cmd = [quantize_bin, gguf_f16, gguf_q4, "Q4_K_M"]
    print("Running:", " ".join(quant_cmd))
    subprocess.run(quant_cmd, check=True)
    print(f"Q4_K_M GGUF saved to {gguf_q4}")
    if KEEP_F16:
        print(f"F16 GGUF kept at {gguf_f16} (KEEP_F16=1).")
    else:
        os.remove(gguf_f16)
        print("F16 intermediate removed (set KEEP_F16=1 to keep it).")

# ─── DONE ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Done! Load in LM Studio:")
print(f"  {os.path.join(GGUF_PATH, GGUF_NAME)}")
print()
print("Suggested LM Studio system prompt:")
print('  "You are a Godot 4 GDScript expert. Answer questions about')
print('   Godot 4 engine features, GDScript syntax, and game development.')
print('   Always use Godot 4 / GDScript 4 syntax, never Godot 3."')
print("=" * 60)
