import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

TARGET_LABELS = [
    "emotion_appeal",
    "authority_appeal",
    "polarization",
    "presumption",
    "exaggeration",
    "rhetorical_framing",
]

LEGACY_MAP = {
    "emotion_appeal": "emotional",
    "authority_appeal": "authority",
    "rhetorical_framing": "framing",
}


def build_system_prompt() -> str:
    return (
        "You are a strict political-rhetoric annotator. Return ONLY valid JSON with keys: "
        "emotion_appeal, authority_appeal, polarization, presumption, exaggeration, rhetorical_framing, rationale. "
        "Each label must be 0 or 1. rationale must be one short sentence.\n\n"
        "Definitions:\n"
        "emotion_appeal: Language designed to evoke strong emotions rather than neutral factual info.\n"
        "authority_appeal: References to institutions, laws, experts, or leaders used to legitimize a claim.\n"
        "polarization: Us-vs-them framing between opposing groups/camps.\n"
        "presumption: Treating claims as obvious/unquestionable without argument.\n"
        "exaggeration: Overstatement beyond literal plausibility (e.g., absolute/apocalyptic claims).\n"
        "rhetorical_framing: Strategic metaphor/analogy/narrative framing shaping interpretation.\n\n"
        "Rules:\n"
        "- Label 1 only if explicit evidence appears in the sentence itself.\n"
        "- If uncertain, prefer 0.\n"
        "- Return only JSON, no markdown."
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("JSON response is not an object")
    return parsed


def normalize_labels(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in TARGET_LABELS:
        value = payload.get(key, 0)
        try:
            normalized[key] = 1 if int(value) == 1 else 0
        except (ValueError, TypeError):
            normalized[key] = 0
    rationale = payload.get("rationale", "")
    normalized["rationale"] = str(rationale).strip()
    return normalized


def annotate_text(
    text: str,
    endpoint: str,
    model: str,
    timeout: int,
    retries: int,
    temperature: float,
) -> dict[str, Any]:
    system_prompt = build_system_prompt()
    payload = {
        "model": model,
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sentence: {text}"},
        ],
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            parsed = extract_json_object(content)
            return normalize_labels(parsed)
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                break

    raise RuntimeError(f"Failed to annotate after {retries + 1} attempts: {last_error}")


def apply_column_strategy(df: pd.DataFrame, use_legacy_columns: bool) -> pd.DataFrame:
    for label in TARGET_LABELS:
        if label not in df.columns:
            df[label] = ""
        df[label] = df[label].astype("string")
    if "rationale" not in df.columns:
        df["rationale"] = ""
    df["rationale"] = df["rationale"].astype("string")

    if use_legacy_columns:
        for _, legacy in LEGACY_MAP.items():
            if legacy not in df.columns:
                df[legacy] = ""
            df[legacy] = df[legacy].astype("string")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate dataset rows with an Ollama model.")
    parser.add_argument("--input", default="dataset/dataset.csv", help="Input dataset CSV path")
    parser.add_argument("--output", default="dataset/dataset_annotated.csv", help="Output CSV path")
    parser.add_argument("--model", default="gpt-oss:20b", help="Ollama model name")
    parser.add_argument("--endpoint", default="http://localhost:11434/api/chat", help="Ollama chat endpoint")
    parser.add_argument("--text-column", default="text", help="Text column to annotate")
    parser.add_argument("--start-row", type=int, default=0, help="Start row index (0-based)")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to annotate; 0 means all")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between requests")
    parser.add_argument(
        "--use-legacy-columns",
        action="store_true",
        help="Also mirror values into legacy columns: emotional/authority/framing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call Ollama; write zeros and empty rationale for selected rows",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        raise SystemExit(f"Text column not found: {args.text_column}")

    df = apply_column_strategy(df, args.use_legacy_columns)

    start = max(args.start_row, 0)
    end = len(df)
    if args.max_rows > 0:
        end = min(end, start + args.max_rows)

    if start >= end:
        raise SystemExit("No rows selected for annotation. Check --start-row and --max-rows.")

    selected_indices = list(range(start, end))

    for row_index in tqdm(selected_indices, desc="Annotating"):
        text = str(df.at[row_index, args.text_column]).strip()
        if not text:
            continue

        if args.dry_run:
            labels = {key: 0 for key in TARGET_LABELS}
            labels["rationale"] = ""
        else:
            labels = annotate_text(
                text=text,
                endpoint=args.endpoint,
                model=args.model,
                timeout=args.timeout,
                retries=args.retries,
                temperature=args.temperature,
            )

        for key in TARGET_LABELS:
            df.at[row_index, key] = str(labels[key])
        df.at[row_index, "rationale"] = labels["rationale"]

        if args.use_legacy_columns:
            for source, target in LEGACY_MAP.items():
                df.at[row_index, target] = str(labels[source])

        if args.sleep > 0:
            time.sleep(args.sleep)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Annotated rows: {len(selected_indices)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
