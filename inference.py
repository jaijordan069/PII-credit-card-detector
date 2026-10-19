"""
inference.py — Standalone Credit Card Number Detector
======================================================
Loads the trained DistilBERT student model and runs the full
detection pipeline: regex extraction → Luhn gate → context scoring
→ student model → combined score → verdict.

Usage:
    python inference.py

Prerequisites:
    pip install torch transformers
    Trained model directory: ./card_detector_model/
"""

import re
import os
import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = "./card_detector_model"

CARD_PATTERN = re.compile(r'\b(?:\d[ -]?){12,18}\d\b')

POSITIVE_KEYWORDS = [
    "credit card", "debit card", "card number", "visa", "mastercard",
    "amex", "american express", "discover", "charge", "billing",
    "payment", "cvv", "expiry", "expires", "exp ", "checkout",
]

NEGATIVE_KEYWORDS = [
    "invoice", "order", "tracking", "serial", "employee", "reference",
    "policy", "batch", "ticket", "document", "account number",
    "phone", "zip", "postal",
]

DETECTION_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Lazy singletons — model loaded once on first call
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None
_device = None


def _load_model():
    """Load the DistilBERT student model from MODEL_DIR (once)."""
    global _tokenizer, _model, _device

    if _model is not None:
        return

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory '{MODEL_DIR}' not found. "
            "Run Cell 12 in the notebook to generate it."
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    _model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.to(_device)
    _model.eval()


# ---------------------------------------------------------------------------
# Luhn check
# ---------------------------------------------------------------------------

def luhn_check(number: str) -> bool:
    """
    Validate a digit string using the Luhn algorithm.

    Args:
        number: Raw candidate string (may contain spaces/dashes).

    Returns:
        True if the number passes the Luhn check, False otherwise.
    """
    digits = [int(d) for d in number if d.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = d * 2 if i % 2 == 1 else d
        if n > 9:
            n -= 9
        total += n
    return total % 10 == 0


def is_suspiciously_patterned(number: str) -> bool:
    """
    Reject trivially patterned numbers (all same digit, arithmetic
    sequences, palindromes) that pass Luhn by construction.

    Args:
        number: Raw candidate string.

    Returns:
        True if the pattern is suspiciously regular.
    """
    digits = [d for d in number if d.isdigit()]
    digit_str = ''.join(digits)
    if len(set(digits)) <= 2:
        return True
    if digit_str == digit_str[::-1]:
        return True
    diffs = [int(digits[i + 1]) - int(digits[i]) for i in range(len(digits) - 1)]
    if len(set(diffs)) == 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Context keyword scorer
# ---------------------------------------------------------------------------

def context_score(text: str, candidate_pos: int, window: int = 80) -> dict:
    """
    Score the text context around a candidate number using keyword lists.

    Scans a ±window character window for positive (payment) and negative
    (invoice/reference) keywords and returns a composite boost value.

    Args:
        text:          Full input text.
        candidate_pos: Character index where the candidate starts.
        window:        Half-width of the context window in characters.

    Returns:
        Dict with context_boost, positive_keywords_found, negative_keywords_found.
    """
    start = max(0, candidate_pos - window)
    end = min(len(text), candidate_pos + window)
    context = text[start:end].lower()

    pos_hits = [kw for kw in POSITIVE_KEYWORDS if kw in context]
    neg_hits = [kw for kw in NEGATIVE_KEYWORDS if kw in context]

    boost = 0.0
    boost += min(len(pos_hits) * 0.10, 0.30)
    boost -= min(len(neg_hits) * 0.08, 0.24)

    return {
        "context_boost": round(boost, 4),
        "positive_keywords_found": pos_hits,
        "negative_keywords_found": neg_hits,
    }


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------

def predict(text: str, threshold: float = DETECTION_THRESHOLD) -> dict:
    """
    Run the full credit card detection pipeline on input text.

    Pipeline:
        1. Regex extraction of digit sequences (12–19 digits)
        2. Luhn check + pattern gate (hard cap if invalid)
        3. DistilBERT student model → P(card)
        4. Context keyword scoring (±boost)
        5. Score fusion → combined score
        6. Threshold gate → verdict

    Args:
        text:      Input text to analyse.
        threshold: Combined score threshold for positive detection (default 0.50).

    Returns:
        Dict with verdict, student_score, candidates, card_hits, detected_count.
    """
    _load_model()

    # --- Student model inference ---
    enc = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.no_grad():
        output = _model(**enc)
        student_probs = torch.softmax(output.logits, dim=-1).cpu()
        student_score = student_probs[0][1].item()

    # --- Candidate extraction and scoring ---
    candidates = [m.group() for m in CARD_PATTERN.finditer(text)]
    card_hits = []

    for candidate in candidates:
        pos = text.find(candidate)
        luhn_valid = luhn_check(candidate) and not is_suspiciously_patterned(candidate)
        ctx = context_score(text, pos)

        if luhn_valid:
            combined = (
                0.50 * student_score
                + 0.50 * (ctx["context_boost"] + 1.0) / 1.5
            )
        else:
            combined = min(student_score * 0.20, 0.10)

        card_hits.append({
            "candidate": candidate,
            "luhn_valid": luhn_valid,
            "student_score": round(student_score, 4),
            "combined": round(combined, 4),
            "detected": combined >= threshold,
            "keywords": ctx["positive_keywords_found"],
        })

    detected = [h for h in card_hits if h["detected"]]

    return {
        "text": text,
        "student_score": round(student_score, 4),
        "candidates": len(candidates),
        "card_hits": card_hits,
        "detected_count": len(detected),
        "verdict": "🚨 CARD DETECTED" if detected else "✅ CLEAN",
    }


# ---------------------------------------------------------------------------
# CLI output helper
# ---------------------------------------------------------------------------

def _pretty_print(result: dict) -> None:
    """Print a formatted detection result to stdout."""
    print("\n" + "=" * 65)
    print(f"TEXT    : {result['text'][:80]}")
    print(f"VERDICT : {result['verdict']}")
    print(f"Student P(card): {result['student_score']}")
    if result["card_hits"]:
        print("CANDIDATES:")
        for h in result["card_hits"]:
            status = "🚨 DETECTED" if h["detected"] else "✅ clean"
            print(f"  {h['candidate']}")
            print(f"    Luhn    : {h['luhn_valid']}")
            print(f"    Student : {h['student_score']}")
            print(f"    Combined: {h['combined']}  →  {status}")
            if h["keywords"]:
                print(f"    Keywords: {h['keywords']}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        # True positives
        "Please charge my credit card 2739581640293810, expiry 08/27.",
        "My Visa 4532-0151-1283-0366 was declined at checkout.",
        # False positive candidates — invoice / reference numbers
        "Invoice #6264488694611952 is due by end of month.",
        # Named entity penalty
        "John Smith's reference number is 2739581640293810.",
        # Patterned number — should fail gate
        "Card 4111-1111-1111-1111 appeared in the system log.",
    ]

    print("Credit Card Detector — Inference Test")
    print(f"Model: {MODEL_DIR}")

    for text in samples:
        result = predict(text)
        _pretty_print(result)
