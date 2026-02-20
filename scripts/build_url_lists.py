import argparse
from collections import OrderedDict
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.presidency.ucsb.edu"

EXCLUDE_PREFIXES = {
    "/documents/app-categories",
    "/documents/presidential-documents-archive-guidebook",
}


def is_document_link(href: str) -> bool:
    if not href.startswith("/documents/"):
        return False
    return not any(href.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def extract_document_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    container = soup.select_one(".view-content") or soup
    links: list[str] = []
    for anchor in container.find_all("a", href=True):
        href = anchor["href"].strip()
        if is_document_link(href):
            links.append(urljoin(BASE_URL, href))
    return links


def find_next_page(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    next_link = soup.find("a", string=lambda s: s and "next" in s.lower())
    if next_link and next_link.get("href"):
        return urljoin(BASE_URL, next_link["href"])
    return None


def crawl_category(category_url: str, limit: int) -> list[str]:
    results: list[str] = []
    seen = set()
    url = category_url
    while url and len(results) < limit:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        links = extract_document_links(response.text)
        for link in links:
            if link not in seen:
                seen.add(link)
                results.append(link)
                if len(results) >= limit:
                    break
        url = find_next_page(response.text)
    return results


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    ordered = OrderedDict.fromkeys(items)
    return list(ordered.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build URL lists from APP categories.")
    parser.add_argument("--presidential-limit", type=int, default=80)
    parser.add_argument("--congressional-limit", type=int, default=30)
    args = parser.parse_args()

    presidential_categories = [
        "https://www.presidency.ucsb.edu/documents/app-categories/spoken-addresses-and-remarks/presidential/state-the-union-addresses",
        "https://www.presidency.ucsb.edu/documents/app-categories/spoken-addresses-and-remarks/presidential/inaugural-addresses",
        "https://www.presidency.ucsb.edu/documents/app-categories/elections-and-transitions/campaign-documents",
        "https://www.presidency.ucsb.edu/documents/app-categories/elections-and-transitions/convention-speeches",
        "https://www.presidency.ucsb.edu/documents/app-categories/elections-and-transitions/debates",
    ]

    congressional_category = "https://www.presidency.ucsb.edu/documents/app-categories/congressional"

    presidential_links: list[str] = []
    for category_url in presidential_categories:
        presidential_links.extend(crawl_category(category_url, args.presidential_limit))
        if len(presidential_links) >= args.presidential_limit:
            break

    presidential_links = unique_preserve_order(presidential_links)[: args.presidential_limit]

    congressional_links = crawl_category(congressional_category, args.congressional_limit)

    print("# presidential")
    for link in presidential_links:
        print(link)

    print("\n# congressional")
    for link in congressional_links:
        print(link)


if __name__ == "__main__":
    main()
