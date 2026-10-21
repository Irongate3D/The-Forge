"""
queue.py — Forge Job Queue & Checkpoint Engine
===============================================
Persistent job state machine. Every stage completion is written to disk.
Crash anywhere, restart, pipeline resumes from last checkpoint.

Job stages (in order):
  discovered → scraped → chunked → formatted → validated → complete
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Valid stages in order
STAGES = ["discovered", "scraped", "chunked", "formatted", "validated", "complete"]
VALID_STATUSES = ["pending", "running", "partial", "done", "error", "skipped"]

SOURCE_TYPES = ["url", "pdf", "epub", "mp4", "zip", "text"]


class Job:
    """Represents a single source item moving through the pipeline."""

    def __init__(self, data: dict):
        self._data = data

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self._data["id"]

    @property
    def source_type(self) -> str:
        return self._data["type"]

    @property
    def source(self) -> str:
        return self._data["source"]

    @property
    def domain(self) -> str:
        return self._data["domain"]

    @property
    def stages(self) -> dict:
        return self._data["stages"]

    @property
    def pairs_generated(self) -> int:
        return self._data.get("pairs_generated", 0)

    @property
    def error(self) -> Optional[str]:
        return self._data.get("error")

    @property
    def output_file(self) -> Optional[str]:
        return self._data.get("output_file")

    @property
    def is_complete(self) -> bool:
        return self._data["stages"]["complete"]["status"] == "done"

    @property
    def has_error(self) -> bool:
        return any(s["status"] == "error" for s in self._data["stages"].values())

    def get_stage(self, stage: str) -> dict:
        return self._data["stages"][stage]

    def stage_status(self, stage: str) -> str:
        return self._data["stages"][stage]["status"]

    # ── Mutators ───────────────────────────────────────────────────────────

    def set_stage(self, stage: str, status: str, **kwargs):
        """Update a stage status and optional metadata."""
        assert stage in STAGES, f"Unknown stage: {stage}"
        assert status in VALID_STATUSES, f"Unknown status: {status}"
        self._data["stages"][stage].update({
            "status": status,
            "ts": datetime.now().isoformat(),
            **kwargs
        })

    def set_output_file(self, path: str):
        self._data["output_file"] = path

    def add_pairs(self, count: int):
        self._data["pairs_generated"] = self._data.get("pairs_generated", 0) + count

    def set_error(self, message: str, stage: str):
        self._data["error"] = message
        self.set_stage(stage, "error", message=message)
        log.error(f"[Job {self.id}] Error at stage {stage}: {message}")

    def clear_error(self):
        self._data["error"] = None

    def to_dict(self) -> dict:
        return self._data.copy()

    def resume_stage(self) -> Optional[str]:
        """Return the stage this job should resume from, or None if complete."""
        for stage in STAGES:
            status = self._data["stages"][stage]["status"]
            if status in ("pending", "partial", "running", "error"):
                return stage
        return None  # All done

    def __repr__(self):
        resume = self.resume_stage()
        return f"<Job {self.id} type={self.source_type} resume_at={resume} pairs={self.pairs_generated}>"


class JobQueue:
    """
    Persistent queue backed by jobs.json.
    Thread-safe via file-level locking (single-process use assumed).
    """

    def __init__(self, domain_path: Path):
        self.domain_path = domain_path
        self.queue_dir = domain_path / "queue"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file = self.queue_dir / "jobs.json"
        self._jobs: dict[str, dict] = {}
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        if self.jobs_file.exists():
            try:
                self._jobs = json.loads(self.jobs_file.read_text(encoding="utf-8"))
                log.info(f"Loaded {len(self._jobs)} jobs from queue")
            except Exception as e:
                log.error(f"Failed to load jobs.json: {e} — starting fresh")
                self._jobs = {}
        else:
            self._jobs = {}

    def _save(self):
        """Atomic write — write to temp then rename."""
        tmp = self.jobs_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._jobs, indent=2), encoding="utf-8")
        tmp.replace(self.jobs_file)

    def save_job(self, job: Job):
        """Persist a single job's current state."""
        self._jobs[job.id] = job.to_dict()
        self._save()

    # ── Job creation ───────────────────────────────────────────────────────

    def _make_stages(self) -> dict:
        return {stage: {"status": "pending"} for stage in STAGES}

    def add_url(self, url: str, domain: str) -> Job:
        # Deduplicate by source
        existing = self._find_by_source(url)
        if existing:
            log.info(f"URL already queued: {url} ({existing.id})")
            return existing
        job_id = f"web_{uuid.uuid4().hex[:8]}"
        stages = self._make_stages()
        stages["discovered"]["status"] = "done"
        stages["discovered"]["ts"] = datetime.now().isoformat()
        data = {
            "id": job_id,
            "type": "url",
            "source": url,
            "domain": domain,
            "added": datetime.now().isoformat(),
            "stages": stages,
            "pairs_generated": 0,
            "output_file": None,
            "error": None,
            "meta": {}
        }
        job = Job(data)
        self.save_job(job)
        log.info(f"Queued URL: {url} → {job_id}")
        return job

    def add_file(self, path: Path, source_type: str, domain: str) -> Job:
        existing = self._find_by_source(str(path))
        if existing:
            log.info(f"File already queued: {path} ({existing.id})")
            return existing
        prefix = source_type[:3]
        job_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
        stages = self._make_stages()
        stages["discovered"]["status"] = "done"
        stages["discovered"]["ts"] = datetime.now().isoformat()
        data = {
            "id": job_id,
            "type": source_type,
            "source": str(path),
            "domain": domain,
            "added": datetime.now().isoformat(),
            "stages": stages,
            "pairs_generated": 0,
            "output_file": None,
            "error": None,
            "meta": {}
        }
        job = Job(data)
        self.save_job(job)
        log.info(f"Queued {source_type}: {path.name} → {job_id}")
        return job

    # ── Queries ────────────────────────────────────────────────────────────

    def _find_by_source(self, source: str) -> Optional[Job]:
        for data in self._jobs.values():
            if data["source"] == source:
                return Job(data)
        return None

    def get_job(self, job_id: str) -> Optional[Job]:
        data = self._jobs.get(job_id)
        return Job(data) if data else None

    def pending_jobs(self) -> list[Job]:
        """Return jobs that need work, in order added."""
        jobs = []
        for data in self._jobs.values():
            job = Job(data)
            if not job.is_complete and not job.has_error:
                jobs.append(job)
        jobs.sort(key=lambda j: j._data["added"])
        return jobs

    def all_jobs(self) -> list[Job]:
        return [Job(d) for d in self._jobs.values()]

    def error_jobs(self) -> list[Job]:
        return [Job(d) for d in self._jobs.values() if Job(d).has_error]

    def status_summary(self) -> dict:
        all_j = self.all_jobs()
        complete = [j for j in all_j if j.is_complete]
        errors = [j for j in all_j if j.has_error]
        pending = [j for j in all_j if not j.is_complete and not j.has_error]
        total_pairs = sum(j.pairs_generated for j in all_j)
        return {
            "total": len(all_j),
            "complete": len(complete),
            "pending": len(pending),
            "errors": len(errors),
            "total_pairs": total_pairs
        }
