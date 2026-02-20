import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def filename_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] or "speech"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", slug).strip("_")
    return f"{slug.lower()}.txt"


def read_urls(paths: list[Path]) -> list[str]:
    urls: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        urls.extend([line.strip() for line in lines if line.strip() and not line.strip().startswith("#")])
    return urls


def extract_speaker(soup: BeautifulSoup) -> str:
    people_links = soup.select('a[href*="/people/"]')
    for link in people_links:
        text = link.get_text(strip=True)
        if text:
            return text

    title = soup.select_one("h1")
    if title:
        title_text = title.get_text(strip=True)
        match = re.search(r"\bby\s+([^\n]+)$", title_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return ""


def extract_year(soup: BeautifulSoup) -> str:
    date_selectors = [
        ".doc-date",
        ".field--name-field-docs-date",
        ".field-docs-date",
        ".date-display-single",
        ".field--name-field-date",
    ]
    date_text = ""
    for selector in date_selectors:
        node = soup.select_one(selector)
        if node:
            date_text = node.get_text(" ", strip=True)
            break

    if not date_text:
        body_text = soup.get_text("\n", strip=True)
        date_match = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+(19|20)\d{2}\b",
            body_text,
        )
        if date_match:
            date_text = date_match.group(0)

    match = re.search(r"(19|20)\d{2}", date_text)
    return match.group(0) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract speaker/year metadata from APP URLs.")
    parser.add_argument(
        "--urls",
        nargs="+",
        default=["data/urls/presidential_urls.txt", "data/urls/congressional_urls.txt"],
        help="One or more URL list files",
    )
    parser.add_argument("--out", default="data/metadata/speeches_metadata.csv")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--user-agent", default="Mozilla/5.0")
    args = parser.parse_args()

    url_paths = [Path(path) for path in args.urls]
    urls = read_urls(url_paths)
    if not urls:
        raise SystemExit("No URLs found to extract metadata.")

    rows: list[dict] = []
    headers = {"User-Agent": args.user_agent}

    for url in tqdm(urls, desc="Metadata"):
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        rows.append(
            {
                "file_name": filename_from_url(url),
                "url": url,
                "speaker": extract_speaker(soup),
                "year": extract_year(soup),
            }
        )

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved metadata for {len(df)} URLs to {out_path}")


if __name__ == "__main__":
    main()
