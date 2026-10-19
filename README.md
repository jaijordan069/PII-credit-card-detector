# PII Credit Card Number Detector

> A multi-signal NLP pipeline for detecting credit card numbers in unstructured text, using GLiNER and BERT NER as teacher models to train a lightweight DistilBERT classifier via knowledge distillation.

---

## Overview

This project implements a production-grade credit card number detection pipeline that combines rule-based validation, named entity recognition, and transformer-based sequence classification. Rather than relying on a single model, the system fuses multiple independent signals — Luhn arithmetic, zero-shot NER confidence, contextual keyword scoring, and person/org entity penalties — into a calibrated combined score. A DistilBERT student model is trained via knowledge distillation from GLiNER and BERT NER teachers, enabling fast, accurate inference without calling heavyweight models at runtime.

---

## Architecture

```
TRAINING PHASE
──────────────────────────────────────────────────────────────
Raw Text ──► Regex Extraction ──► Luhn + Pattern Gate
                                          │
                    ┌─────────────────────┴──────────────────────┐
                    ▼                                             ▼
            GLiNER Teacher                               BERT NER Teacher
      (zero-shot NER confidence)                   (person/org penalty -0.15)
                    │                                             │
                    └─────────────────────┬──────────────────────┘
                                          ▼
                                  Context Keyword Scorer
                                  (80-char window, ±boost)
                                          │
                                          ▼
                                    Score Fusion
                             Luhn(0.35) + GLiNER + Context + Penalty
                                          │
                                          ▼
                            Knowledge Distillation (T=2.0, α=0.5)
                                          │
                                          ▼
                              DistilBERT Student Model
                              (distilbert-base-uncased)

INFERENCE PHASE
──────────────────────────────────────────────────────────────
Raw Text ──► Regex Extraction ──► Luhn Gate ──► DistilBERT Student
                                                       │
                                               Context Keyword Scorer
                                                       │
                                               Combined Score ≥ 0.50?
                                               YES ──► 🚨 CARD DETECTED
                                               NO  ──► ✅ CLEAN
```

---

## Models

| Model | Role | Used At |
|-------|------|---------|
| `urchade/gliner_medium-v2.1` | Zero-shot NER teacher — scores P(credit card number) per candidate | Training only |
| `dslim/bert-base-NER` | Person/Org detection teacher — applies -0.15 penalty when PER/ORG entity found near candidate | Training only |
| `distilbert-base-uncased` | Student classifier — full sentence context classifier, P(card) | Training + Inference |

---

## Signal Fusion

| Signal | Weight / Range | Purpose |
|--------|---------------|----------|
| Luhn check | +0.35 (pass) / 0.00 (fail) | Hard arithmetic gate — eliminates non-card numbers |
| GLiNER confidence | Variable (0.10–1.00) | Zero-shot semantic confidence from teacher |
| Context keyword boost | +0.00 to +0.30 | Positive payment keywords near candidate |
| Context keyword suppress | -0.00 to -0.24 | Negative keywords (invoice, tracking, serial…) |
| Person/Org penalty | -0.15 | BERT NER detects human/org name near candidate |
| **Detection threshold** | **≥ 0.50** | Combined score gate for final verdict |

**Score fusion formula:**
```
If Luhn valid:
    combined = 0.50 × student_score + 0.50 × (context_boost + 1.0) / 1.5

If Luhn invalid:
    combined = min(student_score × 0.20, 0.10)   # hard cap
```

---

## Results

```
              precision    recall  f1-score   support

       clean     1.0000    1.0000    1.0000        60
        card     1.0000    1.0000    1.0000        60

    accuracy                         1.0000       120
   macro avg     1.0000    1.0000    1.0000       120
weighted avg     1.0000    1.0000    1.0000       120

Confusion Matrix:
  TN=60  FP=0
  FN=0   TP=60
```

Training: 800 samples | 5 epochs | Val accuracy: 100%

---

## Quickstart

### Run in Google Colab
Open `credit_card_detector.ipynb` and run all cells top to bottom.

### Standalone inference
```bash
pip install torch transformers
python inference.py
```

> **Note:** `inference.py` requires the saved model directory `./card_detector_model/`. Run Cell 12 in the notebook first to generate it, or download `card_detector_model.zip` from the Colab output.

---

## Design Decisions

- **Luhn as a hard gate, not just a signal** — Any 16-digit sequence can appear in text (invoice IDs, tracking numbers, serial numbers). Luhn check eliminates ~90% of false candidates before any ML runs, keeping latency low and precision high. A failed Luhn caps the combined score at 0.10 regardless of model confidence.

- **Knowledge distillation over direct teacher inference** — Calling GLiNER + BERT NER at inference time adds 200–800ms per request and requires loading two additional models. Distillation transfers the teachers' decision boundaries into DistilBERT during training, so inference runs with a single lightweight model at <20ms.

- **80-character context window** — Card numbers are typically referenced within a single clause. An 80-char window captures the immediately surrounding language (keyword type, entity names) without picking up unrelated sentence context that would add noise.

- **DistilBERT over full BERT** — DistilBERT retains ~97% of BERT's performance at 40% the size and 60% the inference speed. For a binary classification task on short payment-context sentences, the capability gap is negligible and the operational gain is significant.

---

## File Structure

```
PII-credit-card-detector/
├── credit_card_detector.ipynb   # Full training pipeline notebook (13 cells)
├── inference.py                 # Standalone inference script
└── README.md                    # This file
```

---

## Author

**Jay Sharma**  
Senior Professional Services Consultant  
