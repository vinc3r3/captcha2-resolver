"""Stage 1 — collect captcha images from a web page.

How it works (the part worth knowing):
  1. GET the page with a fresh `requests.Session` (each session = a new captcha; the server
     ties the captcha to the session cookie, so reusing a session returns the same image).
  2. Parse the HTML and find the captcha element by a CSS selector.
  3. The image is not a separate URL: it is inlined as a base64 data URI, either
     in `<img src="data:image/jpeg;base64,...">` or in the element's inline
     `style="background: url('data:image/jpeg;base64,...')"`. Plain `src` URLs are also
     supported (fetched with the same session, so the cookie matches).
  4. Decode and save as `<YYYYmmdd_HHMMSS>.jpg`; skip exact duplicates by content hash.
  5. Sleep so each iteration takes at least `interval` seconds; errors are logged and the
     loop continues. Restartable: it counts what's already in the output folder.

The source page is configured through the environment (CAPTCHA_SOURCE_URL, CAPTCHA_SELECTOR)
and intentionally not documented in this repository.
"""

from __future__ import annotations

import base64
import hashlib
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

DATA_URI_RE = re.compile(r"data:image/[\w.+-]+;base64,([A-Za-z0-9+/=\s]+)")
USER_AGENT = "Mozilla/5.0 (compatible; captcha2-resolver-collector/0.1)"


@dataclass
class CollectConfig:
    url: str
    selector: str
    out_dir: Path
    target: int = 2000
    interval: float = 5.0
    timeout: float = 30.0


def extract_image(html: str, selector: str, base_url: str = "") -> tuple[str, bytes | str] | None:
    """Find the captcha in `html`. Returns ("bytes", data) for inline images,
    ("url", absolute_url) for a linked image, or None if not found."""
    from bs4 import BeautifulSoup

    el = BeautifulSoup(html, "html.parser").select_one(selector)
    if el is None:
        return None
    # Look at the element itself and its descendants: <img src>, then inline styles.
    candidates = [el, *el.find_all(True)]
    for node in candidates:
        src = node.get("src") or ""
        m = DATA_URI_RE.search(src)
        if m:
            return "bytes", base64.b64decode(re.sub(r"\s", "", m.group(1)))
        m = DATA_URI_RE.search(node.get("style") or "")
        if m:
            return "bytes", base64.b64decode(re.sub(r"\s", "", m.group(1)))
    for node in candidates:
        if node.name == "img" and node.get("src"):
            return "url", urljoin(base_url, node["src"])
    return None


def _next_path(out_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path, i = out_dir / f"{stamp}.jpg", 1
    while path.exists():  # two captchas within the same second
        path, i = out_dir / f"{stamp}_{i}.jpg", i + 1
    return path


def fetch_one(cfg: CollectConfig, seen: set[str]) -> Path | None:
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    html = session.get(cfg.url, timeout=cfg.timeout).text
    found = extract_image(html, cfg.selector, cfg.url)
    if found is None:
        dump = cfg.out_dir.parent / "page_dump.html"  # gitignored; inspect to fix the selector
        dump.write_text(html, encoding="utf-8")
        raise RuntimeError(f"captcha element not found with selector {cfg.selector!r}; page saved to {dump}")
    kind, payload = found
    data = payload if kind == "bytes" else session.get(payload, timeout=cfg.timeout).content
    digest = hashlib.sha1(data).hexdigest()
    if digest in seen:
        return None
    seen.add(digest)
    path = _next_path(cfg.out_dir)
    path.write_bytes(data)
    return path


def run(cfg: CollectConfig) -> int:
    from tqdm import tqdm

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cfg.out_dir.glob("*.jpg"))
    seen = {hashlib.sha1(p.read_bytes()).hexdigest() for p in existing}
    print(f"{len(existing)} images already in {cfg.out_dir}")
    with tqdm(total=cfg.target, initial=min(len(existing), cfg.target), unit="img") as bar:
        while bar.n < cfg.target:
            start = time.monotonic()
            try:
                if fetch_one(cfg, seen) is not None:
                    bar.update(1)
                else:
                    tqdm.write("duplicate image, skipped")
            except Exception:
                tqdm.write(traceback.format_exc())
            time.sleep(max(0.0, cfg.interval - (time.monotonic() - start)))
    return bar.n
