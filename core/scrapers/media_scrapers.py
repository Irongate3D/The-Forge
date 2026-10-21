"""
mp4_scraper.py — Video Transcription Scraper
=============================================
Uses Whisper (already cached in WSL2) to transcribe MP4 files,
then chunks the transcript for formatting.
"""

import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

CHUNK_SIZE = 600   # words — slightly smaller for spoken content
CHUNK_OVERLAP = 60
WHISPER_MODEL = "large-v3"  # already cached in WSL2


class MP4Scraper:

    def __init__(self, job, queue, pause_flag: Path):
        self.job = job
        self.queue = queue
        self.pause_flag = pause_flag

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 100:
                chunks.append(chunk.strip())
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def run(self) -> list[str]:
        job = self.job
        meta = job._data.setdefault("meta", {})

        if job.stage_status("scraped") == "done":
            return meta.get("scraped_chunks", [])

        mp4_path = Path(job.source)
        if not mp4_path.exists():
            job.set_error(f"MP4 not found: {mp4_path}", "scraped")
            self.queue.save_job(job)
            return []

        log.info(f"[{job.id}] Transcribing: {mp4_path.name}")
        job.set_stage("scraped", "running")
        self.queue.save_job(job)

        # Extract audio to wav first
        wav_path = mp4_path.with_suffix(".wav")
        try:
            subprocess.run([
                "ffmpeg", "-i", str(mp4_path),
                "-ar", "16000", "-ac", "1", "-f", "wav",
                str(wav_path), "-y"
            ], check=True, capture_output=True)
        except Exception as e:
            job.set_error(f"ffmpeg failed: {e}", "scraped")
            self.queue.save_job(job)
            return []

        # Transcribe with Whisper
        try:
            import whisper
            model = whisper.load_model(WHISPER_MODEL)
            result = model.transcribe(str(wav_path))
            transcript = result["text"]
        except Exception as e:
            job.set_error(f"Whisper failed: {e}", "scraped")
            self.queue.save_job(job)
            if wav_path.exists():
                wav_path.unlink()
            return []

        # Clean up wav
        if wav_path.exists():
            wav_path.unlink()

        # Clean transcript
        transcript = re.sub(r"\s+", " ", transcript).strip()
        chunks = self._chunk_text(transcript)

        meta["scraped_chunks"] = chunks
        meta["transcript_length"] = len(transcript)
        job.set_stage("scraped", "done", chunks=len(chunks))
        job.set_stage("chunked", "done", count=len(chunks))
        self.queue.save_job(job)
        log.info(f"[{job.id}] Transcription complete: {len(chunks)} chunks")
        return chunks


"""
zip_scraper.py — ZIP Code Examples Extractor
============================================
Extracts code files from ZIP archives, formats them as
(filename + code) chunks ready for pair generation.
"""

import zipfile
import logging
from pathlib import Path

log = logging.getLogger(__name__)

CODE_EXTENSIONS = {".gd", ".py", ".js", ".ts", ".cs", ".cpp",
                   ".h", ".shader", ".glsl", ".json", ".cfg", ".tres", ".tscn"}
MAX_FILE_SIZE = 50_000  # chars — skip huge generated files


class ZipScraper:

    def __init__(self, job, queue, pause_flag: Path):
        self.job = job
        self.queue = queue
        self.pause_flag = pause_flag

    def run(self) -> list[str]:
        job = self.job
        meta = job._data.setdefault("meta", {})

        if job.stage_status("scraped") == "done":
            return meta.get("scraped_chunks", [])

        zip_path = Path(job.source)
        if not zip_path.exists():
            job.set_error(f"ZIP not found: {zip_path}", "scraped")
            self.queue.save_job(job)
            return []

        log.info(f"[{job.id}] Extracting ZIP: {zip_path.name}")
        job.set_stage("scraped", "running")
        self.queue.save_job(job)

        chunks = []
        files_done: set = set(meta.get("files_done", []))

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                code_files = [n for n in names
                              if Path(n).suffix.lower() in CODE_EXTENSIONS]
                log.info(f"[{job.id}] Found {len(code_files)} code files in ZIP")

                for name in code_files:
                    if name in files_done:
                        continue

                    if self.pause_flag.exists():
                        log.info(f"[{job.id}] Paused mid-ZIP")
                        meta["files_done"] = list(files_done)
                        meta["scraped_chunks"] = chunks
                        job.set_stage("scraped", "partial", files_done=len(files_done))
                        self.queue.save_job(job)
                        return chunks

                    try:
                        content = zf.read(name).decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    if len(content) > MAX_FILE_SIZE:
                        continue

                    filename = Path(name).name
                    ext = Path(name).suffix.lower()

                    # Format as a labelled chunk: filename + code
                    chunk = f"File: {filename}\n\n{content.strip()}"
                    chunks.append(chunk)
                    files_done.add(name)

        except Exception as e:
            job.set_error(str(e), "scraped")
            self.queue.save_job(job)
            return []

        meta["scraped_chunks"] = chunks
        meta["files_done"] = list(files_done)
        job.set_stage("scraped", "done", files=len(files_done), chunks=len(chunks))
        job.set_stage("chunked", "done", count=len(chunks))
        self.queue.save_job(job)
        log.info(f"[{job.id}] ZIP complete: {len(files_done)} files, {len(chunks)} chunks")
        return chunks
