"""
validator.py — Training Pair Quality Validator
===============================================
Filters out low-quality pairs before they go into training.
Fast heuristic checks — no model calls needed.
"""

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Minimum lengths
MIN_INSTRUCTION_WORDS = 5
MIN_RESPONSE_WORDS = 15
MAX_INSTRUCTION_WORDS = 200
MAX_RESPONSE_WORDS = 800

# Phrases that indicate Devstral got confused or hallucinated meta-commentary
BAD_PHRASES = [
    "i cannot", "i can't", "i don't have access",
    "as an ai", "as a language model",
    "i'm sorry", "i apologize",
    "here are some training pairs",
    "here is a training pair",
    "sure, here",
    "certainly!",
    "of course!",
]

# Minimum ratio of response length to instruction length
# Prevents "What is X?" → "X." type pairs
MIN_RESPONSE_RATIO = 2.0


class PairValidator:

    def __init__(self, job, queue):
        self.job = job
        self.queue = queue

    def _score_pair(self, pair: dict) -> tuple[bool, str]:
        """Return (is_valid, reason)."""
        instruction = pair.get("instruction", "").strip()
        response = pair.get("response", "").strip()

        if not instruction or not response:
            return False, "empty field"

        instr_words = len(instruction.split())
        resp_words = len(response.split())

        if instr_words < MIN_INSTRUCTION_WORDS:
            return False, f"instruction too short ({instr_words} words)"

        if resp_words < MIN_RESPONSE_WORDS:
            return False, f"response too short ({resp_words} words)"

        if instr_words > MAX_INSTRUCTION_WORDS:
            return False, f"instruction too long ({instr_words} words)"

        if resp_words > MAX_RESPONSE_WORDS:
            return False, f"response too long ({resp_words} words)"

        if resp_words / max(instr_words, 1) < MIN_RESPONSE_RATIO:
            return False, "response too short relative to instruction"

        combined = (instruction + " " + response).lower()
        for phrase in BAD_PHRASES:
            if phrase in combined:
                return False, f"bad phrase: '{phrase}'"

        # Check instruction ends with ? or is a command (imperative)
        # This is a soft check — we don't reject, just note
        # (some valid instructions are statements like "Explain X")

        # Check for obvious truncation (response ends mid-sentence without punctuation)
        if response and response[-1] not in ".!?`\"')}]":
            # Allow code blocks that end with backticks or closing braces
            last_char = response.strip()[-1]
            if last_char not in ".!?`})\"]":
                log.debug(f"Possible truncation: response ends with '{last_char}'")

        return True, "ok"

    def validate_file(self, jsonl_path: Path) -> Path:
        """
        Read a JSONL file, filter out bad pairs, write to a new _validated.jsonl file.
        Returns path to validated file.
        """
        job = self.job

        if job.stage_status("validated") == "done":
            validated_path = jsonl_path.with_stem(jsonl_path.stem + "_validated")
            log.info(f"[{job.id}] Validation already complete")
            return validated_path

        if not jsonl_path.exists():
            job.set_error(f"JSONL not found: {jsonl_path}", "validated")
            self.queue.save_job(job)
            return jsonl_path

        log.info(f"[{job.id}] Validating pairs in {jsonl_path.name}")

        total = 0
        passed = 0
        rejected_reasons: dict = {}
        validated_path = jsonl_path.with_stem(jsonl_path.stem + "_validated")

        with open(jsonl_path, "r", encoding="utf-8") as fin, \
             open(validated_path, "w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    pair = json.loads(line)
                    valid, reason = self._score_pair(pair)
                    if valid:
                        fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        passed += 1
                    else:
                        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                except json.JSONDecodeError:
                    rejected_reasons["json_error"] = rejected_reasons.get("json_error", 0) + 1

        pass_rate = (passed / total * 100) if total > 0 else 0
        log.info(f"[{job.id}] Validation: {passed}/{total} pairs passed ({pass_rate:.1f}%)")
        if rejected_reasons:
            log.info(f"[{job.id}] Rejection reasons: {rejected_reasons}")

        job.set_stage("validated", "done",
                      total=total,
                      passed=passed,
                      pass_rate=round(pass_rate, 1),
                      rejected_reasons=rejected_reasons)
        self.queue.save_job(job)
        return validated_path
