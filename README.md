# The Forge

A local, crash-resumable pipeline that turns documents, web pages, PDFs, videos and code archives into instruction/response training pairs for fine-tuning a domain-specialist language model.

Part of the Irongate toolset: free, local, private, no subscriptions. Everything runs on your own machine against a local model served by LM Studio. No cloud APIs.

## What it does

```
discovered -> scraped -> chunked -> formatted -> validated -> complete
```

1. **Ingest** from a per-domain watch folder: URLs (`urls.txt`), PDFs, MP4s, ZIPs of code, and plain text/markdown.
2. **Chunk** the extracted text into overlapping word windows.
3. **Format**: each chunk is sent to a local model (Devstral via LM Studio) with a domain-specific prompt that asks for 2-4 instruction/response pairs as JSON.
4. **Validate** with fast heuristic checks (length bounds, response-to-instruction ratio, meta-commentary phrases) and write a `_validated.jsonl` file.

Every stage writes its state to disk (`queue/jobs.json`), so a crash, a pause or a reboot resumes from the last checkpoint instead of starting over.

## Sources supported

| Source | How it is handled |
|---|---|
| Web pages | Crawls same-domain sub-pages (capped), strips navigation and scripts, keeps content tags |
| PDFs | Page-by-page text extraction with `pdfplumber`, checkpointed every 20 pages |
| MP4 video | `ffmpeg` extracts audio, Whisper `large-v3` transcribes it |
| ZIP archives | Extracts code files (`.gd`, `.py`, `.js`, `.cs`, `.tscn` and others) as filename + code chunks |
| Text / Markdown | Read and chunked directly |

## Domains

A domain is a folder under `domains/<name>/` with its own watch folder, queue, datasets and logs. Formatter prompts are defined in `core/formatter.py` (`DOMAIN_PROMPTS`) for `godot`, `nz_law` and a `default`. Add your own by adding a prompt entry and creating the domain folder.

```
domains/godot/
  watch/{pdfs,mp4s,zips,text}/   drop sources here
  watch/urls.txt                 one URL per line, # for comments
  queue/jobs.json                job state and checkpoints
  datasets/                      generated .jsonl (and _validated.jsonl)
  done/                          processed source files
  logs/                          one log per run
```

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai) serving a chat model with the OpenAI-compatible API
- `ffmpeg` on PATH (MP4 only)
- `pip install -r requirements.txt`

## Configuration

Settings are read from environment variables, with defaults matching the author's WSL2 + LM Studio setup:

| Variable | Default | Purpose |
|---|---|---|
| `FORGE_ROOT` | the folder containing `pipeline.py` | Where `domains/` and `pause.flag` live |
| `LM_STUDIO_URL` | `http://172.19.176.1:12345/v1/chat/completions` | Chat completions endpoint (set this to your own; LM Studio's default is `http://localhost:1234/v1/chat/completions`) |
| `FORGE_MODEL` | `lmstudio/mistralai/devstral-small-2-2512` | Model identifier sent to LM Studio |
| `FORGE_VENV_PYTHON` | `/home/irongate/unsloth-env/bin/python3` | Tray app only: Python used inside WSL2 |
| `FORGE_PIPELINE_PY` | `/mnt/d/Irongate/Forge/pipeline.py` | Tray app only: pipeline path inside WSL2 |

Requests are sent with `stream: false`. Streaming breaks tool calls on AMD ROCm in the author's setup, so the formatter never streams.

## Usage

```bash
# Drop sources into domains/godot/watch/, then:
python pipeline.py --domain godot

python pipeline.py --domain godot --status          # queue summary
python pipeline.py --domain godot --retry-errors    # reset failed jobs and retry
```

To pause safely, create a file named `pause.flag` in the Forge root. The pipeline stops between chunks and resumes later from its checkpoint. Delete the file to continue.

### Windows tray app

`forge_tray.py` is a system-tray controller (run, pause, stop, retry, switch domain, open folders) that launches the pipeline inside WSL2. It needs `pystray` and `pillow` installed in Windows Python.

## Scope and limitations

- This repo covers **training-data generation only**. Fine-tuning (Unsloth LoRA on ROCm) and GGUF conversion are separate and not included here.
- Quality control is heuristic. There is no model-based grading and no semantic deduplication in this repo.
- The web crawler does not read `robots.txt` yet and the queue assumes a single process.
- Output quality depends on the local model you use for formatting. Review samples before training on them.

## Responsible use

Only ingest content you have the right to use. Respect website terms and `robots.txt`, and check the licence of any documentation, code or media you turn into training data. Generated pairs can inherit the restrictions of their source material.

## Licence

MIT. See `LICENSE`.
