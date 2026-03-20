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
        "You are an annotation assistant for a Natural Language Processing research project.\n\n"
        "Your task is to label ONE political speech segment according to six rhetorical strategies.\n\n"
        "This is a MULTI-LABEL classification task.\n"
        "Each label must be either 0 (absent) or 1 (present).\n\n"
        "IMPORTANT RULES\n"
        "- Only label a strategy as 1 if it is clearly expressed in the text itself.\n"
        "- If the sentence is factual, administrative, descriptive, or neutral → all labels should be 0.\n"
        "- Do NOT infer meaning from broader context.\n"
        "- Be conservative: if unsure, choose 0.\n"
        "- Focus on language use, not political ideology or whether the statement is true.\n\n"
        "Labels and definitions:\n\n"
        "emotion_appeal\n"
        "Language designed to evoke emotions such as fear, hope, pride, sympathy, gratitude, anger, or reassurance.\n"
        "Examples:\n"
        "\"Our children deserve a better future.\"\n"
        "\"We are grateful to those who serve.\"\n"
        "\"This crisis is hurting families.\"\n\n"
        "authority_appeal\n"
        "Language that relies on experts, institutions, legal principles, or official authorities to support a claim.\n"
        "Examples:\n"
        "\"According to the World Health Organization...\"\n"
        "\"The Constitution requires...\"\n"
        "\"The OPCW confirmed...\"\n\n"
        "Note: mentioning an institution without using it as support is NOT authority_appeal.\n\n"
        "polarization\n"
        "Language creating an \"us vs them\" division or conflict between groups.\n"
        "Examples:\n"
        "\"They are taking advantage of us.\"\n"
        "\"We must stand together against those who threaten our way of life.\"\n\n"
        "presumption\n"
        "A claim presented as obvious, necessary, or universally accepted without evidence.\n"
        "Examples:\n"
        "\"We must act now.\"\n"
        "\"This clearly requires action.\"\n"
        "\"The problem demands immediate reform.\"\n\n"
        "exaggeration\n"
        "Hyperbole or absolute language exaggerating scale, certainty, or universality.\n"
        "Examples:\n"
        "\"Everyone knows.\"\n"
        "\"Never before.\"\n"
        "\"Each and every.\"\n"
        "\"The greatest threat.\"\n\n"
        "rhetorical_framing\n"
        "Strategic wording that shapes how an issue is interpreted using value-based language or narrative framing.\n"
        "Examples:\n"
        "\"protect our families\"\n"
        "\"defend freedom\"\n"
        "\"economic fairness\"\n"
        "\"engines of destruction\"\n\n"
        "If the sentence only reports facts, statistics, or policy details, label rhetorical_framing = 0.\n\n"
        "Decision priority rules:\n\n"
        "1. If the sentence is neutral or factual → all labels = 0.\n"
        "2. emotion_appeal requires clear emotional language.\n"
        "3. authority_appeal requires explicit reliance on authority.\n"
        "4. polarization requires explicit group conflict.\n"
        "5. presumption requires a normative claim presented as obvious.\n"
        "6. exaggeration requires clear hyperbolic wording.\n"
        "7. rhetorical_framing requires value-laden interpretation language.\n\n"
        "Return ONLY valid JSON in this format:\n\n"
        "{\n"
        "  \"emotion_appeal\": 0,\n"
        "  \"authority_appeal\": 0,\n"
        "  \"polarization\": 0,\n"
        "  \"presumption\": 0,\n"
        "  \"exaggeration\": 0,\n"
        "  \"rhetorical_framing\": 0\n"
        "}\n\n"
        "Output ONLY valid JSON and nothing else."
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
            {"role": "user", "content": f"Now label this segment:\n\nTEXT:\n{text}"},
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
    if "rationale" in df.columns:
        df = df.drop(columns=["rationale"])

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
        help="Do not call Ollama; write zeros for selected rows",
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
