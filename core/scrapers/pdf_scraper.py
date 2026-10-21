"""
pdf_scraper.py — PDF Text Extractor with Checkpoint Support
============================================================
Extracts text from PDF files page by page, chunks into
training-ready segments.
"""

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import pdfplumber
except ImportError:
    log.warning("pdfplumber not installed — run: pip install pdfplumber")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80


class PDFScraper:

    def __init__(self, job, queue, pause_flag: Path):
        self.job = job
        self.queue = queue
        self.pause_flag = pause_flag

    def _is_paused(self) -> bool:
        return self.pause_flag.exists()

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
            log.info(f"[{job.id}] PDF extract already complete")
            return meta.get("scraped_chunks", [])

        pdf_path = Path(job.source)
        if not pdf_path.exists():
            job.set_error(f"PDF not found: {pdf_path}", "scraped")
            self.queue.save_job(job)
            return []

        pages_done: int = meta.get("pages_done", 0)
        all_chunks: list = meta.get("scraped_chunks", [])

        log.info(f"[{job.id}] Extracting PDF: {pdf_path.name} (resuming from page {pages_done})")
        job.set_stage("scraped", "running")
        self.queue.save_job(job)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                full_text_parts = []

                for i, page in enumerate(pdf.pages):
                    if i < pages_done:
                        continue

                    if self._is_paused():
                        log.info(f"[{job.id}] Paused at page {i}")
                        meta["pages_done"] = i
                        meta["scraped_chunks"] = all_chunks
                        job.set_stage("scraped", "partial", pages_done=i, total=total_pages)
                        self.queue.save_job(job)
                        return all_chunks

                    text = page.extract_text() or ""
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        full_text_parts.append(text)

                    # Checkpoint every 20 pages
                    if i % 20 == 0 and i > 0:
                        meta["pages_done"] = i
                        job.set_stage("scraped", "partial", pages_done=i, total=total_pages)
                        self.queue.save_job(job)

                full_text = "\n\n".join(full_text_parts)
                all_chunks = self._chunk_text(full_text)

        except Exception as e:
            job.set_error(str(e), "scraped")
            self.queue.save_job(job)
            log.error(f"[{job.id}] PDF extraction failed: {e}")
            return []

        meta["scraped_chunks"] = all_chunks
        meta["pages_done"] = total_pages
        job.set_stage("scraped", "done", pages=total_pages, chunks=len(all_chunks))
        job.set_stage("chunked", "done", count=len(all_chunks))
        self.queue.save_job(job)
        log.info(f"[{job.id}] PDF complete: {total_pages} pages, {len(all_chunks)} chunks")
        return all_chunks
