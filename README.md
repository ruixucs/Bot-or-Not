# Bot-or-Not: Social Media Bot Detection System

A bot detection system for the Bot-or-Not challenge, consisting of two core modules:

1. **`max.py`** — Main detection pipeline (ML + rule-based post-processing)
2. **`add_engagement_bots.py`** — Supplementary detection script (Engagement Farming + Encoding Anomalies)

---

## Architecture

```
Dataset (JSON) ──> max.py (147 features + Stacking Ensemble)
                      │
                      ├── Model prediction (XGBoost + LightGBM + CatBoost → LogisticRegression)
                      ├── Threshold optimization (Competition Score: +4 TP, -1 FN, -2 FP)
                      └── Rule-based post-processing (R1: confused persona, R4: same-second burst)
                      │
                      ▼
              detections.{lang}.txt
                      │
                      ▼
              add_engagement_bots.py (regex rule scan)
                      │
                      ├── Engagement Farming detection
                      ├── Hex encoding leak detection (LLM generation artifact)
                      └── JSON format leak detection (pipeline artifact)
                      │
                      ▼
              Final detection list
```

---

## max.py — Main Detection Pipeline

### Overview

A multi-layer bot detection pipeline that extracts **147 features**, classifies users with a **Stacking Ensemble** model, and applies rule-based post-processing to rescue missed bots.

### Feature Engineering (147 features)

| Layer | Feature Group | Count | Description |
|-------|---------------|-------|-------------|
| Layer 1 | User Profile | ~15 | Username, bio, location and other profile signals |
| Layer 1 | Posting Behavior | ~20 | Posting frequency, intervals, hour entropy, burst detection |
| Layer 1 | Content | ~20 | Text length, vocabulary, hashtags, URLs, duplicate ratio |
| Layer 1 | Advanced Temporal | ~20 | Time DNA, inter-arrival distribution fitting, session analysis, second-level periodicity |
| Layer 1 | Text Stylometry | ~14 | N-gram repetition, Jaccard similarity, Zipf deviation, compression ratio |
| Layer 1 | Mention Network | ~5 | Mention entropy, in-dataset ratio, concentration, reciprocity |
| Layer 2 | Sentence Embeddings | ~5 | `paraphrase-multilingual-MiniLM-L12-v2` embedding similarity stats |
| Layer 2 | Perplexity | ~7 | GPT-2 (EN) / BLOOM-560m (FR) language model perplexity |
| Layer 2 | Topic Coherence | ~8 | Sequential coherence, topic diversity, topic switch rate |
| Layer 2 | Sentiment Analysis | ~9 | Multilingual sentiment polarity, volatility, extremity |
| Layer 3 | Cross-user | ~4 | Cross-user embedding similarity |
| Layer 3 | HDBSCAN Clustering | ~8 | Embedding + temporal pattern clustering |

### Model Architecture

**Stacking Ensemble:**

- **Base Models (Layer 1)**: XGBoost + LightGBM + CatBoost
  - Custom asymmetric loss functions (W_POS=5.0, W_NEG=2.0) encoding the competition scoring directly
- **Meta-Learner (Layer 2)**: LogisticRegressionCV
  - Trained on out-of-fold predictions from base models
- **Hyperparameter Optimization**: Optuna (TPE sampler, 200 trials, 3-fold CV)

### Threshold Optimization

Two-pass search strategy:
1. Coarse pass (0.05 ~ 0.96, step=0.05)
2. Fine pass (best ± 0.08, step=0.002)

Objective: Competition Score = `+4 × TP − 1 × FN − 2 × FP`

### Rule-Based Post-Processing

Re-examines users predicted as "human" by the main model:

| Rule | Condition | Target |
|------|-----------|--------|
| **R1** | `question_ratio >= 0.8` + `consecutive_streak >= 0.6` | "Confused persona" bots (nearly all posts are questions) |
| **R4** | `max_same_timestamp >= 4` (4+ posts at the exact same second) | Batch-posting bots |

### Datasets

| Dataset | Language | Purpose |
|---------|----------|---------|
| Dataset 30 | English | Practice (cross-validation) |
| Dataset 31 | French | Practice (cross-validation) |
| Dataset 32 | English | Practice (cross-validation) |
| Dataset 33 | French | Practice (cross-validation) |
| Dataset 34 | English | Evaluation |
| Dataset 35 | French | Evaluation |

### Output

```
runs/<RUN_NAME>/
├── <RUN_NAME>.log                         # Full execution log
├── <RUN_NAME>_summary.json                # Machine-readable results summary
├── hyperparams/                           # Hyperparameter snapshot
├── Bot_killer_RX.detections.en.txt        # English detections (one user ID per line)
├── Bot_killer_RX.detections.fr.txt        # French detections
└── Bot_killer_RX.detections.all.txt       # Combined detections
```

### Dependencies

```
numpy, pandas, scipy, scikit-learn
xgboost, lightgbm, catboost
optuna, hdbscan
sentence-transformers, transformers, torch
```

### Usage

```bash
python max.py
```

---

## add_engagement_bots.py — Supplementary Detection Script

### Overview

Performs a supplementary scan over `max.py` detection results, using regex-based rules to catch three types of bots the main model may miss:

1. **Engagement Farming** — Accounts soliciting likes, retweets, and follows
2. **Hex Encoding Leaks** — LLM-generated text with encoding errors
3. **JSON Format Leaks** — Pipeline artifacts where raw JSON was not unpacked

### Engagement Farming Rules

| Rule | Condition | Description |
|------|-----------|-------------|
| **R-EF1** | `sol_count >= 4` | High-frequency solicitation (4+ posts with "like and retweet", etc.) |
| **R-EF2** | `sol_count >= 2` + `sol_ratio >= 10%` | Moderate solicitation with significant ratio |
| **R-EF3** | `bio_match` + `sol_count >= 1` | Bio contains "follow back" + any solicitation post |
| **R-EF4** | `bio_match` + `total <= 12` | Bio contains "follow back" + low activity |
| **R-EF5** | `max_repeat >= 3` + `sol_count >= 1` | Repetitive template + solicitation (e.g. "Jour X pour que...") |

Supports both English and French solicitation pattern matching.

### Hex / Encoding Anomaly Rules

| Rule | Condition | Description |
|------|-----------|-------------|
| **R-HEX1** | `hex_accent_posts >= 2` or `hex_accent_hits >= 3` | Posts with hex accent leaks (e.g. `cine9ma` = cinéma) |
| **R-HEX2** | `hex_emoji_posts >= 2` | Posts with raw hex emoji codes (e.g. `01f627a0`) |
| **R-JSON** | `json_leak_posts >= 1` | Posts with JSON array format leak (e.g. `['text']`) |

**False-positive safeguards:**
- `HEX_ACCENT_CODES` only includes digit-containing hex codes (`e9`, `e8`, `e0`, `f4`, `e7`, `f9`, `c3`); purely alphabetic codes (`ea`, `ee`, etc.) are excluded to avoid matching common English words
- `COMMON_HEX_WORDS` filter list excludes English words composed entirely of hex characters (`decade`, `beefed`, etc.)
- Pure numeric strings (dates, large numbers) are automatically excluded

### Usage

```bash
python add_engagement_bots.py
```

Currently runs in **Dry Run mode**: prints detection results without writing to files.

### Configuration

Edit the constants at the top of the file:

```python
DATASET_FILES = {
    "en": "dataset.posts&users.34.json",
    "fr": "dataset.posts&users.35.json",
}
DETECTION_DIR = os.path.join("runs", "max_v3_20260214_131317")
```

