"""
web_scraper.py — URL Scraper with Checkpoint Support
=====================================================
Crawls a URL (and optionally sub-pages on the same domain),
extracts clean text, splits into chunks.

Checkpoint: scraped page URLs stored in job meta so crashes
            resume from last completed page.
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

log = logging.getLogger(__name__)

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    log.warning("requests/beautifulsoup4 not installed — run: pip install requests beautifulsoup4")


CHUNK_SIZE = 800        # target words per chunk
CHUNK_OVERLAP = 80      # word overlap between chunks
MAX_PAGES = 200         # safety cap for crawls
CRAWL_DELAY = 0.5       # seconds between requests (polite)

# Tags to strip entirely (scripts, nav, ads, footers etc)
STRIP_TAGS = ["script", "style", "nav", "footer", "header",
              "aside", "form", "button", "iframe", "noscript"]

# Tags whose text content is useful
CONTENT_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                "pre", "code", "td", "th", "blockquote", "dt", "dd"]


class WebScraper:

    def __init__(self, job, queue, pause_flag: Path):
        self.job = job
        self.queue = queue
        self.pause_flag = pause_flag
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; IrongateForge/1.0; research scraper)"
        })

    def _is_paused(self) -> bool:
        return self.pause_flag.exists()

    def _clean_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from a BeautifulSoup page."""
        for tag in soup(STRIP_TAGS):
            tag.decompose()
        parts = []
        for tag in soup.find_all(CONTENT_TAGS):
            text = tag.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 30:  # skip tiny fragments
                parts.append(text)
        return "\n\n".join(parts)

    def _same_domain_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Find all same-domain links on a page."""
        base_domain = urlparse(base_url).netloc
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
                # Strip fragment
                clean = full.split("#")[0].rstrip("/")
                if clean:
                    links.add(clean)
        return list(links)

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a single page, return soup or None on error."""
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            log.warning(f"Failed to fetch {url}: {e}")
            return None

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping word chunks."""
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > 100:  # skip tiny chunks
                chunks.append(chunk.strip())
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def run(self) -> list[str]:
        """
        Scrape the job's URL. Returns list of text chunks.
        Resumes from checkpoint if partially complete.
        """
        job = self.job
        meta = job._data.setdefault("meta", {})

        # Restore checkpoint state
        scraped_urls: set = set(meta.get("scraped_urls", []))
        all_chunks: list = meta.get("scraped_chunks", [])
        to_visit: list = meta.get("to_visit", [job.source])

        if job.stage_status("scraped") == "done":
            log.info(f"[{job.id}] Scrape already complete — {len(all_chunks)} chunks")
            return all_chunks

        log.info(f"[{job.id}] Starting scrape of {job.source}")
        log.info(f"[{job.id}] Resuming: {len(scraped_urls)} pages done, {len(to_visit)} queued")

        job.set_stage("scraped", "running")
        self.queue.save_job(job)

        visited_this_run = 0

        while to_visit and len(scraped_urls) < MAX_PAGES:
            if self._is_paused():
                log.info(f"[{job.id}] Pause flag detected — saving checkpoint")
                meta["scraped_urls"] = list(scraped_urls)
                meta["scraped_chunks"] = all_chunks
                meta["to_visit"] = to_visit
                job.set_stage("scraped", "partial",
                              pages_done=len(scraped_urls),
                              chunks=len(all_chunks))
                self.queue.save_job(job)
                return all_chunks

            url = to_visit.pop(0)
            if url in scraped_urls:
                continue

            soup = self._fetch_page(url)
            if soup is None:
                scraped_urls.add(url)  # mark as attempted
                continue

            text = self._clean_text(soup)
            chunks = self._chunk_text(text)
            all_chunks.extend(chunks)
            scraped_urls.add(url)
            visited_this_run += 1

            # Only crawl sub-pages for the root domain (not external)
            if visited_this_run == 1:
                sub_links = self._same_domain_links(soup, url)
                new_links = [l for l in sub_links if l not in scraped_urls]
                to_visit.extend(new_links[:MAX_PAGES])
                log.info(f"[{job.id}] Found {len(new_links)} sub-pages to crawl")

            # Checkpoint every 10 pages
            if visited_this_run % 10 == 0:
                meta["scraped_urls"] = list(scraped_urls)
                meta["scraped_chunks"] = all_chunks
                meta["to_visit"] = to_visit
                job.set_stage("scraped", "partial",
                              pages_done=len(scraped_urls),
                              chunks=len(all_chunks))
                self.queue.save_job(job)
                log.info(f"[{job.id}] Checkpoint: {len(scraped_urls)} pages, {len(all_chunks)} chunks")

            time.sleep(CRAWL_DELAY)

        # Complete
        meta["scraped_urls"] = list(scraped_urls)
        meta["scraped_chunks"] = all_chunks
        meta["to_visit"] = []
        job.set_stage("scraped", "done",
                      pages=len(scraped_urls),
                      chunks=len(all_chunks))
        job.set_stage("chunked", "done", count=len(all_chunks))
        self.queue.save_job(job)
        log.info(f"[{job.id}] Scrape complete: {len(scraped_urls)} pages, {len(all_chunks)} chunks")
        return all_chunks
