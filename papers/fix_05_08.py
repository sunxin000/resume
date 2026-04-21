#!/usr/bin/env python3
"""Fix papers 05 and 08 - download images correctly."""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(OUTPUT_DIR)

DATA_URI_PREFIX = "data:"


def download_image(url, path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"  WARN: {url}: {e}")
        return False


def convert(name, html_url):
    print(f"\nProcessing: {name}")
    img_dir = f"{name}_images"
    os.makedirs(img_dir, exist_ok=True)

    r = requests.get(html_url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    title = soup.find("title").get_text().strip()
    content = (
        soup.find("article")
        or soup.find("div", class_="ltx_page_content")
        or soup.find("body")
    )

    base = html_url.rstrip("/") + "/"
    cnt = 0

    for img in content.find_all("img"):
        src = img.get("src", "")
        if not src:
            continue
        if src.startswith(DATA_URI_PREFIX):
            continue
        if "portrait" in src.lower():
            img.decompose()
            continue

        # Strip arxiv version prefix from src path if present
        clean = re.sub(r"^\d{4}\.\d{5}v\d+/", "", src)
        full_url = base + clean

        fname = clean.replace("/", "_")
        if not os.path.splitext(fname)[1]:
            fname += ".png"

        local_path = os.path.join(img_dir, fname)
        rel_path = f"{img_dir}/{fname}"

        if download_image(full_url, local_path):
            img["src"] = rel_path
            cnt += 1
            print(f"  Got: {fname}")
        else:
            img["src"] = rel_path

    for x in content.find_all("nav"):
        x.decompose()
    for x in content.find_all("footer"):
        x.decompose()

    mk = md(
        str(content),
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "link"],
    )
    mk = re.sub(r"\n{4,}", "\n\n\n", mk)
    full_md = f"# {title}\n\n{mk}"

    with open(f"{name}.md", "w", encoding="utf-8") as f:
        f.write(full_md)
    print(f"  Saved: {name}.md ({len(full_md)} chars, {cnt} images)")


convert("05_Multimodal-Adaptive-RAG", "https://arxiv.org/html/2603.00511v1")
time.sleep(1)
convert("08_Uncertainty-Calibration", "https://arxiv.org/html/2303.12973v2")
