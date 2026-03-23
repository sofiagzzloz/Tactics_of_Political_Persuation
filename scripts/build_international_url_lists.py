import argparse
import re
from collections import OrderedDict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def unique_preserve_order(items: list[str]) -> list[str]:
    ordered = OrderedDict.fromkeys(items)
    return list(ordered.keys())


def extract_links(html: str, match: callable, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if match(href):
            links.append(urljoin(base_url, href))
    return links


def crawl_paginated(base_template: str, pages: int, matcher: callable, base_url: str) -> list[str]:
    results: list[str] = []
    for page in range(pages):
        url = base_template.format(page=page)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        results.extend(extract_links(response.text, matcher, base_url))
    return results


def crawl_canada_json(limit: int, page_size: int = 50) -> list[str]:
    speech_type_id = "d1ad84d2-dad7-40e7-84a6-38eaa27b4c36"
    results: list[str] = []
    offset = 0
    headers = {"Accept": "application/vnd.api+json"}
    while len(results) < limit:
        url = (
            "https://www.pm.gc.ca/en/jsonapi/node/article"
            f"?filter[field_news_category.id]={speech_type_id}"
            f"&page[limit]={page_size}&page[offset]={offset}"
        )
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            break
        for item in data:
            alias = item.get("attributes", {}).get("path", {}).get("alias")
            if alias:
                results.append(f"https://www.pm.gc.ca/en{alias}")
                if len(results) >= limit:
                    break
        offset += page_size
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build URL lists for UK/Canada/Australia speeches.")
    parser.add_argument("--uk-pages", type=int, default=50)
    parser.add_argument("--uk-limit", type=int, default=100)
    parser.add_argument("--canada-pages", type=int, default=5)
    parser.add_argument("--canada-limit", type=int, default=100)
    parser.add_argument("--australia-pages", type=int, default=5)
    parser.add_argument("--australia-limit", type=int, default=100)
    args = parser.parse_args()

    uk_template = (
        "https://www.gov.uk/search/news-and-communications?announcement_filter_option=speeches"
        "&order=updated-newest&page={page}"
    )
    uk_match = lambda href: href.startswith("/government/speeches/")
    uk_links = crawl_paginated(uk_template, args.uk_pages, uk_match, "https://www.gov.uk")
    uk_links = unique_preserve_order(uk_links)[: args.uk_limit]

    canada_links = crawl_canada_json(args.canada_limit)
    canada_links = unique_preserve_order(canada_links)[: args.canada_limit]

    australia_template = "https://www.pm.gov.au/media?page={page}"
    australia_match = lambda href: href.startswith("/media/") and "?page=" not in href
    australia_links = crawl_paginated(australia_template, args.australia_pages, australia_match, "https://www.pm.gov.au")
    australia_links = unique_preserve_order(australia_links)[: args.australia_limit]

    print("# uk")
    for link in uk_links:
        print(link)
    print("\n# canada")
    for link in canada_links:
        print(link)
    print("\n# australia")
    for link in australia_links:
        print(link)


if __name__ == "__main__":
    main()
