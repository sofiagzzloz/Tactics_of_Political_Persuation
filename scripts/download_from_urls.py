import argparse
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return text.lower()


def filename_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] or "speech"
    return f"{slugify(slug)}.txt"


def extract_text(html: str, selector: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    if selector:
        node = soup.select_one(selector)
        if node:
            return node.get_text("\n", strip=True)
    for fallback in ["article", "main", "body"]:
        node = soup.select_one(fallback)
        if node:
            return node.get_text("\n", strip=True)
    return soup.get_text("\n", strip=True)


def read_urls(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download speech text from a list of URLs.")
    parser.add_argument("--urls-file", required=True, help="Path to text file containing URLs")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory for raw .txt files")
    parser.add_argument(
        "--selector",
        default="div.field-docs-content",
        help="CSS selector for main content (default: Presidency Project)",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests")
    parser.add_argument("--user-agent", default="Mozilla/5.0", help="User-Agent header")

    args = parser.parse_args()

    urls_path = Path(args.urls_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = read_urls(urls_path)
    if not urls:
        raise SystemExit("No URLs found in the urls file.")

    headers = {"User-Agent": args.user_agent}

    for url in tqdm(urls, desc="Downloading"):
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        text = extract_text(response.text, args.selector)

        out_path = out_dir / filename_from_url(url)
        out_path.write_text(text, encoding="utf-8")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
