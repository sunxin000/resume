#!/usr/bin/env python3
"""Fix papers 05, 07, 08 that failed or were missing."""

import os
import re
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urljoin

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def download_image(img_url, save_path):
    try:
        resp = requests.get(img_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"  [WARN] Failed: {img_url}: {e}")
        return False


def convert_paper(name, html_url, base_img_url=None):
    print(f"\nProcessing: {name}")
    img_dir = os.path.join(OUTPUT_DIR, f"{name}_images")
    os.makedirs(img_dir, exist_ok=True)

    resp = requests.get(html_url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text().strip() if title_tag else name

    content = soup.find("article") or soup.find("div", class_="ltx_page_content") or soup.find("body")

    images = content.find_all("img")
    img_count = 0
    # Use provided base_img_url or derive from html_url
    if base_img_url is None:
        base_img_url = html_url.rstrip("/") + "/"

    for img in images:
        src = img.get("src", "")
        if not src or src.startswith(""):
            continue
        if "portrait" in src.lower():
            img.decompose()
            continue

        # Strip version prefix from src if present (e.g. "2603.00511v1/x1.png" -> "x1.png")
        clean_src = re.sub(r'^\d{4}\.\d{5}v\d+/', '', src)
        full_url = base_img_url + clean_src

        img_filename = clean_src.replace("/", "_")
        if not os.path.splitext(img_filename)[1]:
            img_filename += ".png"
        local_path = os.path.join(img_dir, img_filename)
        rel_path = f"{name}_images/{img_filename}"

        if download_image(full_url, local_path):
            img["src"] = rel_path
            img_count += 1
            print(f"  Downloaded: {img_filename}")
        else:
            img["src"] = rel_path

    for nav in content.find_all("nav"):
        nav.decompose()
    for footer in content.find_all("footer"):
        footer.decompose()

    markdown_content = md(str(content), heading_style="ATX", bullets="-", strip=["script", "style", "link"])
    markdown_content = re.sub(r"\n{4,}", "\n\n\n", markdown_content)
    full_md = f"# {title}\n\n{markdown_content}"

    md_path = os.path.join(OUTPUT_DIR, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_md)

    print(f"  Saved: {name}.md ({len(full_md)} chars, {img_count} images)")


# Fix paper 05 - image URLs need version prefix stripped
convert_paper(
    "05_Multimodal-Adaptive-RAG",
    "https://arxiv.org/html/2603.00511v1",
    base_img_url="https://arxiv.org/html/2603.00511v1/"
)

# Fix paper 08 - use v2 which has HTML
convert_paper(
    "08_Uncertainty-Calibration",
    "https://arxiv.org/html/2303.12973v2",
    base_img_url="https://arxiv.org/html/2303.12973v2/"
)

# Paper 07 - try ACL Anthology
print("\nProcessing: 07_Noise-Robust-Semi-Supervised (ACL Anthology)")
try:
    acl_url = "https://aclanthology.org/2023.findings-emnlp.876/"
    resp = requests.get(acl_url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # ACL Anthology pages are landing pages, not full HTML papers
    # Check if there's an embedded PDF or full text link
    pdf_link = soup.find("a", class_="btn-primary")
    print(f"  ACL Anthology is a landing page only, no full HTML version available.")
    print(f"  Paper 07 PDF was already downloaded. Skipping markdown conversion.")
except Exception as e:
    print(f"  [ERROR]: {e}")

print("\nDone!")
