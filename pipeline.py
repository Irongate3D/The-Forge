"""
pipeline.py — Forge Pipeline Orchestrator
==========================================
Ties everything together. Reads the watch folder, queues new sources,
processes all pending jobs through the full pipeline.

Run directly:
  python pipeline.py --domain godot
  python pipeline.py --domain godot --status
  python pipeline.py --domain godot --retry-errors
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
FORGE_ROOT = Path(os.environ.get("FORGE_ROOT", Path(__file__).resolve().parent))

# ── Logging setup ──────────────────────────────────────────────────────────
def setup_logging(domain_path: Path) -> None:
    log_dir = domain_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ]
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ── Imports after path setup ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.queue import JobQueue
from core.scrapers.web_scraper import WebScraper
from core.scrapers.pdf_scraper import PDFScraper
from core.scrapers.media_scrapers import MP4Scraper, ZipScraper
from core.formatter import DevstralFormatter
from core.validator import PairValidator


# ── Domain config loader ───────────────────────────────────────────────────

def load_domain_config(domain_path: Path) -> dict:
    config_file = domain_path / "config.json"
    if config_file.exists():
        return json.loads(config_file.read_text(encoding="utf-8"))
    return {"domain": domain_path.name}


# ── Watch folder scanner ───────────────────────────────────────────────────

def scan_watch_folder(domain_path: Path, queue: JobQueue, domain_name: str) -> int:
    """Scan watch subfolders and queue any new sources. Returns count queued."""
    watch = domain_path / "watch"
    watch.mkdir(parents=True, exist_ok=True)
    queued = 0

    # URLs from urls.txt
    urls_file = watch / "urls.txt"
    if urls_file.exists():
        lines = urls_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and line.startswith("http"):
                job = queue.add_url(line, domain_name)
                if job.stage_status("discovered") == "done" and \
                        job.stage_status("scraped") == "pending":
                    queued += 1

    # PDFs
    pdf_dir = watch / "pdfs"
    if pdf_dir.exists():
        for f in pdf_dir.glob("*.pdf"):
            job = queue.add_file(f, "pdf", domain_name)
            if job.stage_status("scraped") == "pending":
                queued += 1

    # MP4s
    mp4_dir = watch / "mp4s"
    if mp4_dir.exists():
        for f in mp4_dir.glob("*.mp4"):
            job = queue.add_file(f, "mp4", domain_name)
            if job.stage_status("scraped") == "pending":
                queued += 1

    # ZIPs
    zip_dir = watch / "zips"
    if zip_dir.exists():
        for f in zip_dir.glob("*.zip"):
            job = queue.add_file(f, "zip", domain_name)
            if job.stage_status("scraped") == "pending":
                queued += 1

    # Plain text / markdown
    text_dir = watch / "text"
    if text_dir.exists():
        for ext in ("*.txt", "*.md"):
            for f in text_dir.glob(ext):
                job = queue.add_file(f, "text", domain_name)
                if job.stage_status("scraped") == "pending":
                    queued += 1

    if queued:
        log.info(f"Queued {queued} new sources from watch folder")
    else:
        log.info("No new sources found in watch folder")

    return queued


# ── Per-job processor ──────────────────────────────────────────────────────

def process_job(job, queue: JobQueue, domain_path: Path,
                domain_name: str, pause_flag: Path) -> bool:
    """
    Run a single job through all remaining stages.
    Returns True if completed successfully.
    """
    log.info(f"Processing job {job.id} ({job.source_type}): {job.source[:80]}")

    datasets_dir = domain_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    done_dir = domain_path / "done"
    done_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Scrape ────────────────────────────────────────────────────
    resume = job.resume_stage()
    chunks = []

    if resume in ("scraped", "chunked"):
        source_type = job.source_type
        if source_type == "url":
            scraper = WebScraper(job, queue, pause_flag)
        elif source_type == "pdf":
            scraper = PDFScraper(job, queue, pause_flag)
        elif source_type == "mp4":
            scraper = MP4Scraper(job, queue, pause_flag)
        elif source_type == "zip":
            scraper = ZipScraper(job, queue, pause_flag)
        elif source_type == "text":
            # Simple text file — read and chunk directly
            chunks = _chunk_text_file(job, queue, pause_flag)
        else:
            log.error(f"Unknown source type: {source_type}")
            return False

        if source_type != "text":
            chunks = scraper.run()

        if pause_flag.exists():
            return False  # paused mid-scrape

        if job.has_error:
            log.error(f"Job {job.id} failed at scrape stage")
            return False

    # Reload chunks from meta if scrape was already done
    if not chunks:
        chunks = job._data.get("meta", {}).get("scraped_chunks", [])

    if not chunks:
        log.warning(f"Job {job.id} produced no chunks — marking skipped")
        job.set_stage("formatted", "skipped")
        job.set_stage("validated", "skipped")
        job.set_stage("complete", "done")
        queue.save_job(job)
        _move_to_done(job, done_dir)
        return True

    # ── Stage 2: Format ────────────────────────────────────────────────────
    if job.resume_stage() == "formatted":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = datasets_dir / f"{domain_name}_{job.id}_{ts}.jsonl"
        if job.output_file:
            output_file = Path(job.output_file)
        else:
            job.set_output_file(str(output_file))
            queue.save_job(job)

        formatter = DevstralFormatter(job, queue, pause_flag, domain=domain_name)
        formatter.run(chunks, output_file)

        if pause_flag.exists():
            return False

        if job.has_error:
            log.error(f"Job {job.id} failed at format stage")
            return False

    # ── Stage 3: Validate ──────────────────────────────────────────────────
    if job.resume_stage() == "validated":
        output_file = Path(job.output_file) if job.output_file else None
        if output_file and output_file.exists():
            validator = PairValidator(job, queue)
            validator.validate_file(output_file)
        else:
            log.warning(f"Job {job.id} — no output file to validate")
            job.set_stage("validated", "skipped")
            queue.save_job(job)

    # ── Stage 4: Complete ──────────────────────────────────────────────────
    job.set_stage("complete", "done")
    queue.save_job(job)
    _move_to_done(job, done_dir)
    log.info(f"Job {job.id} complete — {job.pairs_generated} pairs generated")
    return True


def _chunk_text_file(job, queue, pause_flag) -> list[str]:
    """Read a plain text or markdown file and chunk it."""
    from pathlib import Path as P
    path = P(job.source)
    if not path.exists():
        job.set_error(f"Text file not found: {path}", "scraped")
        queue.save_job(job)
        return []
    try:
        text = path.read_text(encoding="utf-8")
        words = text.split()
        CHUNK_SIZE, OVERLAP = 800, 80
        chunks = []
        start = 0
        while start < len(words):
            chunk = " ".join(words[start:start+CHUNK_SIZE])
            if len(chunk.strip()) > 100:
                chunks.append(chunk.strip())
            start += CHUNK_SIZE - OVERLAP
        job.set_stage("scraped", "done", chunks=len(chunks))
        job.set_stage("chunked", "done", count=len(chunks))
        job._data.setdefault("meta", {})["scraped_chunks"] = chunks
        queue.save_job(job)
        return chunks
    except Exception as e:
        job.set_error(str(e), "scraped")
        queue.save_job(job)
        return []


def _move_to_done(job, done_dir: Path):
    """Move a file-based source to the done folder."""
    if job.source_type == "url":
        return  # URLs don't have files to move
    src = Path(job.source)
    if src.exists():
        dest = done_dir / src.name
        try:
            src.rename(dest)
            log.info(f"Moved {src.name} → done/")
        except Exception as e:
            log.warning(f"Could not move {src.name} to done: {e}")


# ── Status display ─────────────────────────────────────────────────────────

def show_status(queue: JobQueue, domain_name: str):
    summary = queue.status_summary()
    print(f"\n{'='*50}")
    print(f"  The Forge — {domain_name.upper()} Domain Status")
    print(f"{'='*50}")
    print(f"  Total jobs:     {summary['total']}")
    print(f"  Complete:       {summary['complete']}")
    print(f"  Pending:        {summary['pending']}")
    print(f"  Errors:         {summary['errors']}")
    print(f"  Total pairs:    {summary['total_pairs']}")
    print(f"{'='*50}")

    pending = queue.pending_jobs()
    if pending:
        print(f"\n  Pending jobs:")
        for job in pending[:10]:
            resume = job.resume_stage()
            src = job.source[:60] + "..." if len(job.source) > 60 else job.source
            print(f"    [{job.source_type:4}] {job.id} | resume: {resume:12} | {src}")
        if len(pending) > 10:
            print(f"    ... and {len(pending)-10} more")

    errors = queue.error_jobs()
    if errors:
        print(f"\n  Error jobs:")
        for job in errors:
            print(f"    {job.id}: {job.error}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="The Forge — Training Data Pipeline")
    parser.add_argument("--domain", default="godot",
                        help="Domain name (matches folder under domains/)")
    parser.add_argument("--status", action="store_true",
                        help="Show queue status and exit")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Reset error jobs to pending and retry")
    parser.add_argument("--forge-root", default=str(FORGE_ROOT),
                        help="Path to Forge root directory")
    args = parser.parse_args()

    forge_root = Path(args.forge_root)
    domain_name = args.domain
    domain_path = forge_root / "domains" / domain_name
    domain_path.mkdir(parents=True, exist_ok=True)

    setup_logging(domain_path)
    log.info(f"The Forge starting — domain: {domain_name}")
    log.info(f"Forge root: {forge_root}")

    pause_flag = forge_root / "pause.flag"
    queue = JobQueue(domain_path)

    if args.status:
        show_status(queue, domain_name)
        return

    if args.retry_errors:
        errors = queue.error_jobs()
        for job in errors:
            job.clear_error()
            resume = job.resume_stage() or "scraped"
            job.set_stage(resume, "pending")
            queue.save_job(job)
        log.info(f"Reset {len(errors)} error jobs to pending")

    # Scan watch folder for new sources
    scan_watch_folder(domain_path, queue, domain_name)

    # Process all pending jobs
    pending = queue.pending_jobs()
    if not pending:
        log.info("No pending jobs — nothing to do")
        show_status(queue, domain_name)
        return

    log.info(f"Processing {len(pending)} pending jobs")
    completed = 0
    failed = 0

    for job in pending:
        if pause_flag.exists():
            log.info("Pause flag detected — stopping pipeline")
            break
        success = process_job(job, queue, domain_path, domain_name, pause_flag)
        if success:
            completed += 1
        else:
            if pause_flag.exists():
                log.info("Pipeline paused mid-job")
                break
            failed += 1

    log.info(f"Run complete — {completed} jobs completed, {failed} failed")
    show_status(queue, domain_name)


if __name__ == "__main__":
    main()
