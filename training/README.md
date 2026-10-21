# Training: Godot specialist (Unsloth LoRA)

`train_godot.py` fine-tunes **Qwen2.5-3B-Instruct** on a Godot 4 / GDScript instruction dataset produced by The Forge, using Unsloth LoRA on an AMD GPU (ROCm) under WSL2.

## Setup used

| | |
|---|---|
| Base model | Qwen2.5-3B-Instruct, loaded in 4-bit from a local folder (no network calls) |
| Method | LoRA, rank 16, alpha 16, all attention and MLP projection layers |
| Training | 3 epochs, learning rate 2e-4, batch size 2 with 4 gradient-accumulation steps (effective batch 8), AdamW 8-bit, max sequence length 2048, seed 42 |
| Prompt format | Alpaca-style `### Instruction / ### Response` template with the tokenizer's EOS token |
| Hardware | Radeon RX 7900 XTX, ROCm 7.1, PyTorch 2.10.0, WSL2 |

## Data

The script reads a JSONL file with `instruction` and `output` fields. The Forge writes `instruction` and `response`, and the script accepts either name. The Godot run used 7,640 unique pairs after de-duplication. The dataset itself is not included in this repository: generate your own with The Forge from sources you have the right to use.

## Running it

```bash
# inside an environment with Unsloth, TRL, datasets and a ROCm (or CUDA) build of PyTorch
python training/train_godot.py
```

Paths can be overridden with environment variables:

| Variable | Default |
|---|---|
| `GODOT_BASE_MODEL` | `/mnt/d/Irongate/models/Qwen2.5-3B-Instruct` |
| `GODOT_DATASET` | `/mnt/d/Irongate/Forge/domains/godot/datasets/godot_final.jsonl` |
| `GODOT_LORA_OUT` | `/mnt/d/Irongate/models/godot-lora` |

Install Unsloth and PyTorch following Unsloth's instructions for your hardware. Use a dedicated environment rather than installing extra packages over a working GPU setup, because other packages can replace the GPU build of PyTorch.

## Results

Training loss fell from about 2.36 to about 0.52 over the run. The adapters were then merged to float16, converted to GGUF and quantised to Q4_K_M with llama.cpp for use in LM Studio. The merge and conversion scripts are not in this repository yet, and the trained weights are not published.

## Known limitations

Manual evaluation of the first model found:

- It handled typed-array syntax correctly.
- It used the Godot 3 signature for `move_and_slide()` instead of the Godot 4 one.
- Signal-connection idioms were only partly right, and generated state machines were poor.
- A placeholder name ("BumpingPig") appeared in unrelated answers, which points to overfitting.

A second run is planned to target these failure modes (cleaner data for the Godot 4 API, and fewer or more varied epochs).

## Notes

- The model was trained with an Alpaca-style prompt template, not Qwen's chat template. When serving it, use a matching prompt template or answers may degrade.
- Review generated training data before training on it. Model-written pairs can contain wrong API calls.
