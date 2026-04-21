#!/usr/bin/env python3
"""Download arxiv HTML papers and convert to markdown with local images."""

import os
import re
import sys
import time
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urljoin

PAPERS = [
    ("01_KBQA-R1", "https://arxiv.org/html/2512.10999v1"),
    ("02_Divide-Then-Align", "https://arxiv.org/html/2505.20871v1"),
    ("03_Predict-the-Retrieval", "https://arxiv.org/html/2601.11443v1"),
    ("04_ToolWeaver", "https://arxiv.org/html/2601.21947v1"),
    ("05_Multimodal-Adaptive-RAG", "https://arxiv.org/html/2603.00511v1"),
    ("06_DIVE", "https://arxiv.org/html/2408.04400v1"),
    # 07 is from ACL Anthology, handled separately
    ("08_Uncertainty-Calibration", "https://arxiv.org/html/2303.12973v1"),
    ("09_Pin-Tuning", "https://arxiv.org/html/2411.01158v1"),
    ("10_GSLB", "https://arxiv.org/html/2310.05174v1"),
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def download_image(img_url, save_path):
    """Download an image to local path."""
    try:
        resp = requests.get(img_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"  [WARN] Failed to download {img_url}: {e}")
        return False


def convert_paper(name, html_url):
    """Convert one arxiv HTML paper to markdown with images."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"URL: {html_url}")

    # Create output dirs
    img_dir = os.path.join(OUTPUT_DIR, f"{name}_images")
    os.makedirs(img_dir, exist_ok=True)

    # Download HTML
    try:
        resp = requests.get(html_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] Failed to download HTML: {e}")
        return False

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text().strip() if title_tag else name

    # Find main content - arxiv uses <article> or <div class="ltx_page_content">
    content = soup.find("article") or soup.find("div", class_="ltx_page_content")
    if not content:
        content = soup.find("body")

    if not content:
        print(f"  [ERROR] Could not find main content")
        return False

    # Process images: download and update src
    images = content.find_all("img")
    img_count = 0
    for img in images:
        src = img.get("src", "")
        if not src or src.startswith("data:"):
            continue
        # Skip portrait/author photos
        if "portrait" in src.lower():
            img.decompose()
            continue

        # Build full URL
        full_url = urljoin(html_url + "/", src)

        # Determine local filename
        img_filename = src.replace("/", "_")
        if not os.path.splitext(img_filename)[1]:
            img_filename += ".png"
        local_path = os.path.join(img_dir, img_filename)
        rel_path = f"{name}_images/{img_filename}"

        # Download
        if download_image(full_url, local_path):
            img["src"] = rel_path
            img_count += 1
            print(f"  Downloaded: {img_filename}")
        else:
            img["src"] = rel_path  # keep local path anyway

    # Remove navigation, footer, bibliography links that clutter markdown
    for nav in content.find_all("nav"):
        nav.decompose()
    for footer in content.find_all("footer"):
        footer.decompose()

    # Convert HTML to markdown
    markdown_content = md(
        str(content),
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "link"],
    )

    # Clean up excessive whitespace
    markdown_content = re.sub(r"\n{4,}", "\n\n\n", markdown_content)

    # Add title header
    full_md = f"# {title}\n\n{markdown_content}"

    # Write markdown file
    md_path = os.path.join(OUTPUT_DIR, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_md)

    print(f"  Saved: {name}.md ({len(full_md)} chars)")
    print(f"  Images: {img_count} downloaded to {name}_images/")
    return True


def main():
    print("Arxiv HTML -> Markdown Converter")
    print(f"Output directory: {OUTPUT_DIR}")

    success = 0
    failed = []

    for name, url in PAPERS:
        if convert_paper(name, url):
            success += 1
        else:
            failed.append(name)
        time.sleep(1)  # Be polite to arxiv

    print(f"\n{'='*60}")
    print(f"Done! {success}/{len(PAPERS)} papers converted.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"\nNote: Paper 07 (Noise-Robust, EMNLP 2023) is not on arxiv HTML.")
    print(f"It was published on ACL Anthology and may not have an HTML version.")


if __name__ == "__main__":
    main()
