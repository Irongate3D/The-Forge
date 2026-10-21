"""
formatter.py — Devstral Pair Formatter
=======================================
Sends text chunks to Devstral via LM Studio (port 12345) and
converts them into instruction/response training pairs.

stream: false — NON-NEGOTIABLE (streaming breaks tool calls on AMD ROCm)

Checkpoint: tracks which chunks have been formatted so crashes
            resume from last completed chunk.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://172.19.176.1:12345/v1/chat/completions")
DEVSTRAL_MODEL = os.environ.get("FORGE_MODEL", "lmstudio/mistralai/devstral-small-2-2512")
TIMEOUT = 120          # seconds per chunk — Devstral can be slow on long chunks
RETRY_LIMIT = 3
RETRY_DELAY = 5        # seconds between retries
BATCH_CHECKPOINT = 5   # save progress every N chunks


# ── Domain-specific formatter prompts ─────────────────────────────────────

DOMAIN_PROMPTS = {
    "godot": """You are a training data generator for a Godot 4 / GDScript specialist AI.

Given a text chunk from Godot documentation, tutorials, or code examples, generate
2-4 high-quality instruction/response training pairs.

Rules:
- Instructions should be questions or tasks a Godot developer would actually ask
- Responses should be accurate, practical, and include GDScript 4 code examples where relevant
- Focus on GDScript 4 syntax (not GDScript 2 / Godot 3)
- Do not make up API calls — only use what the text supports
- Each pair must be self-contained and useful without context

For CODE chunks: generate pairs like "How do I [do what this code does]?" with the code as answer
For DOCS chunks: generate pairs like "What is [concept]?" or "How do I [task]?"

Output ONLY a JSON array, no other text, no markdown fences:
[
  {"instruction": "...", "response": "..."},
  {"instruction": "...", "response": "..."}
]""",

    "nz_law": """You are a training data generator for a New Zealand law specialist AI.

Given a text chunk from NZ legislation, case law, or legal commentary, generate
2-4 high-quality instruction/response training pairs.

Rules:
- Instructions should be questions a NZ lawyer, law student, or citizen would ask
- Responses should cite the relevant Act, section, and subsection where applicable
- Use plain English — avoid unnecessary legalese
- Do not extrapolate beyond what the text states
- Note if something requires professional legal advice

Output ONLY a JSON array, no other text, no markdown fences:
[
  {"instruction": "...", "response": "..."},
  {"instruction": "...", "response": "..."}
]""",

    "default": """You are a training data generator for a specialist AI assistant.

Given the following text chunk, generate 2-4 high-quality instruction/response
training pairs that capture the key knowledge in this text.

Rules:
- Instructions should be natural questions or tasks a user would ask
- Responses should be accurate, complete, and helpful
- Do not make up information not present in the text
- Each pair must be self-contained

Output ONLY a JSON array, no other text, no markdown fences:
[
  {"instruction": "...", "response": "..."},
  {"instruction": "...", "response": "..."}
]"""
}


class DevstralFormatter:

    def __init__(self, job, queue, pause_flag: Path, domain: str = "default"):
        self.job = job
        self.queue = queue
        self.pause_flag = pause_flag
        self.domain = domain
        self.system_prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["default"])

    def _is_paused(self) -> bool:
        return self.pause_flag.exists()

    def _call_devstral(self, chunk: str) -> Optional[list[dict]]:
        """Send one chunk to Devstral, return list of pairs or None on failure."""
        payload = {
            "model": DEVSTRAL_MODEL,
            "stream": False,   # NON-NEGOTIABLE
            "temperature": 0.7,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Generate training pairs from this text:\n\n{chunk}"}
            ]
        }

        for attempt in range(RETRY_LIMIT):
            try:
                resp = requests.post(LM_STUDIO_URL, json=payload,
                                     timeout=TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Strip markdown fences if Devstral adds them despite instructions
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()

                pairs = json.loads(content)
                if not isinstance(pairs, list):
                    log.warning(f"Devstral returned non-list: {type(pairs)}")
                    return None

                # Validate structure
                valid = []
                for p in pairs:
                    if (isinstance(p, dict) and
                            "instruction" in p and "response" in p and
                            len(p["instruction"]) > 10 and len(p["response"]) > 10):
                        valid.append({
                            "instruction": p["instruction"].strip(),
                            "response": p["response"].strip()
                        })
                return valid if valid else None

            except json.JSONDecodeError as e:
                log.warning(f"JSON parse failed (attempt {attempt+1}): {e}")
            except requests.RequestException as e:
                log.warning(f"Request failed (attempt {attempt+1}): {e}")
            except Exception as e:
                log.warning(f"Unexpected error (attempt {attempt+1}): {e}")

            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY)

        log.error(f"All {RETRY_LIMIT} attempts failed for chunk")
        return None

    def run(self, chunks: list[str], output_file: Path) -> int:
        """
        Format all chunks into pairs, write to output_file (JSONL).
        Returns number of pairs generated.
        """
        job = self.job
        meta = job._data.setdefault("meta", {})

        if job.stage_status("formatted") == "done":
            log.info(f"[{job.id}] Formatting already complete")
            return job.pairs_generated

        chunks_done: int = meta.get("chunks_formatted", 0)
        total_chunks = len(chunks)

        log.info(f"[{job.id}] Formatting {total_chunks} chunks "
                 f"(resuming from {chunks_done})")

        job.set_stage("formatted", "running",
                      chunks_done=chunks_done,
                      chunks_total=total_chunks)
        self.queue.save_job(job)

        # Open output file in append mode so we don't lose completed pairs on restart
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pairs_this_run = 0

        with open(output_file, "a", encoding="utf-8") as fh:
            for i, chunk in enumerate(chunks[chunks_done:], start=chunks_done):
                if self._is_paused():
                    log.info(f"[{job.id}] Paused at chunk {i}/{total_chunks}")
                    meta["chunks_formatted"] = i
                    job.set_stage("formatted", "partial",
                                  chunks_done=i,
                                  chunks_total=total_chunks)
                    self.queue.save_job(job)
                    return pairs_this_run

                pairs = self._call_devstral(chunk)
                if pairs:
                    for pair in pairs:
                        fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    fh.flush()
                    job.add_pairs(len(pairs))
                    pairs_this_run += len(pairs)
                else:
                    log.warning(f"[{job.id}] Chunk {i} produced no pairs — skipping")

                # Checkpoint every N chunks
                if (i + 1) % BATCH_CHECKPOINT == 0:
                    meta["chunks_formatted"] = i + 1
                    job.set_stage("formatted", "partial",
                                  chunks_done=i + 1,
                                  chunks_total=total_chunks)
                    self.queue.save_job(job)
                    log.info(f"[{job.id}] Progress: {i+1}/{total_chunks} chunks, "
                             f"{job.pairs_generated} pairs total")

        meta["chunks_formatted"] = total_chunks
        job.set_stage("formatted", "done",
                      chunks_total=total_chunks,
                      pairs=job.pairs_generated)
        self.queue.save_job(job)
        log.info(f"[{job.id}] Formatting complete: {job.pairs_generated} pairs")
        return pairs_this_run
