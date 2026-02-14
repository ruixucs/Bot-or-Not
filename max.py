#!/usr/bin/env python
# coding: utf-8
"""
Bot-or-Not Challenge - Bot Detection System (Enhanced v3)

Multi-layer bot detection pipeline with 147 features:
 1. Basic Features: User profile + posting behavior + content statistics
 2. Advanced Temporal Features: Time DNA + inter-arrival + session + second-level periodicity
 3. Text Stylometry Features: N-gram repetition + Jaccard + Zipf + compression ratio
 4. Mention Network Features: Mention entropy + in-dataset ratio + concentration + reciprocity
 5. Deep NLP Features: Sentence embeddings + multilingual LLM perplexity (GPT-2 EN, BLOOM FR)
 6. Sentiment Analysis Features: Multilingual sentiment polarity + volatility + extremity
 7. Topic Coherence Features: Sequential coherence + topic diversity + topic clustering
 8. Cross-user Features: Inter-user similarity + HDBSCAN clustering
 9. Threshold Optimization: Custom scoring (+4 TP, -1 FN, -2 FP)
10. Stacking Ensemble: XGBoost + LightGBM + CatBoost -> LogisticRegression
"""

# ============================================================
# 0. Logging & Run Management
# ============================================================
import sys
import os
from datetime import datetime

# Generate a unique run ID with timestamp
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"max_v3_{RUN_TIMESTAMP}"

# Create output directories
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", RUN_NAME)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"{RUN_NAME}.log")


class TeeLogger:
    """Tee stdout/stderr to both console and a log file."""
    def __init__(self, filepath, stream):
        self.file = open(filepath, "a", encoding="utf-8", buffering=1)
        self.stream = stream
        self.encoding = getattr(stream, 'encoding', 'utf-8')

    def write(self, msg):
        self.stream.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return hasattr(self.stream, 'isatty') and self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        self.file.close()


# Redirect stdout and stderr to tee logger
sys.stdout = TeeLogger(LOG_FILE, sys.stdout)
sys.stderr = TeeLogger(LOG_FILE, sys.stderr)

print(f"{'='*60}")
print(f"RUN: {RUN_NAME}")
print(f"LOG: {LOG_FILE}")
print(f"DIR: {LOG_DIR}")
print(f"{'='*60}\n")


import json
import re
import zlib
import warnings
from collections import Counter
from functools import partial
from itertools import combinations

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.optimize import curve_fit

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegressionCV
import hdbscan

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.cluster import KMeans

from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')

# ============================================================
# GPU Detection & Configuration
# ============================================================
USE_GPU = torch.cuda.is_available()
print(f"GPU available: {USE_GPU}")
if USE_GPU:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Tree models use CPU: dataset is small (~275 users), CPU training < 1s per model.
# GPU is reserved for NLP models (sentence transformer, perplexity, sentiment)
# which benefit hugely from GPU acceleration.
XGB_DEVICE = 'cpu'
CB_TASK_TYPE = 'CPU'
print(f"Tree model device: XGBoost={XGB_DEVICE}, CatBoost={CB_TASK_TYPE} (CPU is faster for small datasets)")
print(f"NLP model device: {'cuda' if USE_GPU else 'cpu'}")


# ## Section 1: Data Loading

# Auto-detect environment: Colab (Google Drive) or Local
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/bot or not'
    print(f"[Colab] DATA_DIR = {DATA_DIR}")
except ImportError:
    # Local: use script directory
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"[Local] DATA_DIR = {DATA_DIR}")



def load_dataset(json_path, bots_path=None):
    """Load a dataset from JSON and optionally load bot labels."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    posts_df = pd.DataFrame(data['posts'])
    users_df = pd.DataFrame(data['users'])
    metadata = {
        'id': data['id'],
        'lang': data['lang'],
        'metadata': data['metadata']
    }

    # Load bot labels if available (practice datasets)
    bot_ids = set()
    if bots_path and os.path.exists(bots_path):
        with open(bots_path, 'r') as f:
            bot_ids = set(line.strip() for line in f if line.strip())
        users_df['is_bot'] = users_df['id'].isin(bot_ids).astype(int)
    else:
        users_df['is_bot'] = -1  # Unknown

    print(f"Dataset {metadata['id']} ({metadata['lang']}): "
          f"{len(users_df)} users, {len(posts_df)} posts, {len(bot_ids)} known bots")
    return posts_df, users_df, bot_ids, metadata


# Load all practice datasets
datasets = {}
for ds_id in [30, 31, 32, 33]:
    json_path = os.path.join(DATA_DIR, f'dataset.posts&users.{ds_id}.json')
    bots_path = os.path.join(DATA_DIR, f'dataset.bots.{ds_id}.txt')
    if os.path.exists(json_path):
        posts_df, users_df, bot_ids, meta = load_dataset(json_path, bots_path)
        datasets[ds_id] = {
            'posts': posts_df, 'users': users_df,
            'bot_ids': bot_ids, 'meta': meta
        }

print(f"\nLoaded {len(datasets)} datasets: {list(datasets.keys())}")
print(f"English datasets: {[k for k,v in datasets.items() if v['meta']['lang']=='en']}")
print(f"French datasets: {[k for k,v in datasets.items() if v['meta']['lang']=='fr']}")


# ## Section 2: Feature Engineering (Layer 1 - Basic Features)
# 
# Extract 35+ features from user profiles, posting behavior, and content statistics.


EMOJI_PATTERN = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)

# ============================================================
# 2a. User Profile Features
# ============================================================
def extract_user_profile_features(user):
    feats = {}

    # Basic stats (from dataset)
    feats['tweet_count'] = user['tweet_count']
    feats['z_score'] = user['z_score']

    # Username features
    uname = str(user.get('username', ''))
    feats['username_length'] = len(uname)
    feats['username_digit_ratio'] = sum(c.isdigit() for c in uname) / max(len(uname), 1)
    feats['username_underscore_count'] = uname.count('_')
    feats['username_upper_ratio'] = sum(c.isupper() for c in uname) / max(len(uname), 1)
    feats['username_has_numbers'] = int(any(c.isdigit() for c in uname))

    # Name features
    name = str(user.get('name', '') or '')
    feats['name_length'] = len(name)
    feats['name_emoji_count'] = len(EMOJI_PATTERN.findall(name))
    feats['name_word_count'] = len(name.split())

    # Description features
    desc = str(user.get('description', '') or '')
    feats['has_description'] = int(bool(desc.strip()))
    feats['description_length'] = len(desc)
    feats['description_emoji_count'] = len(EMOJI_PATTERN.findall(desc))
    feats['description_word_count'] = len(desc.split()) if desc.strip() else 0
    feats['description_hashtag_count'] = desc.count('#')
    feats['description_url_count'] = len(re.findall(r'https?://\S+', desc))
    feats['description_pipe_count'] = desc.count('|')  # Bio separators like "Gamer | Streamer"

    # Location features
    loc = str(user.get('location', '') or '')
    feats['has_location'] = int(bool(loc.strip()))
    feats['location_length'] = len(loc)

    return feats


# ============================================================
# 2b. Posting Behavior Features
# ============================================================
def extract_behavioral_features(user_posts):
    feats = {}
    n = len(user_posts)

    if n < 2:
        return {k: 0 for k in [
            'posting_frequency', 'time_span_hours',
            'avg_interval', 'std_interval', 'min_interval', 'max_interval',
            'cv_interval', 'median_interval',
            'interval_skewness', 'interval_kurtosis',
            'hour_entropy', 'night_post_ratio', 'morning_post_ratio',
            'evening_post_ratio',
            'burst_count_60s', 'burst_count_300s',
            'unique_hours', 'unique_days',
            'weekend_ratio', 'posts_per_day_std',
            'max_posts_in_hour', 'regularity_score'
        ]}

    timestamps = pd.to_datetime(user_posts['created_at']).sort_values().reset_index(drop=True)

    # Time span
    time_span = (timestamps.max() - timestamps.min()).total_seconds() / 3600
    feats['time_span_hours'] = time_span
    feats['posting_frequency'] = n / max(time_span, 0.01)

    # Time intervals (in seconds)
    intervals = timestamps.diff().dropna().dt.total_seconds().values
    feats['avg_interval'] = np.mean(intervals)
    feats['std_interval'] = np.std(intervals)
    feats['min_interval'] = np.min(intervals)
    feats['max_interval'] = np.max(intervals)
    feats['median_interval'] = np.median(intervals)
    feats['cv_interval'] = np.std(intervals) / max(np.mean(intervals), 0.01)

    # Distribution shape
    if len(intervals) >= 4:
        feats['interval_skewness'] = float(stats.skew(intervals))
        feats['interval_kurtosis'] = float(stats.kurtosis(intervals))
    else:
        feats['interval_skewness'] = 0
        feats['interval_kurtosis'] = 0

    # Hour distribution entropy
    hours = timestamps.dt.hour
    hour_counts = hours.value_counts(normalize=True)
    feats['hour_entropy'] = float(stats.entropy(hour_counts))

    # Time-of-day ratios
    feats['night_post_ratio'] = hours.between(0, 5).mean()
    feats['morning_post_ratio'] = hours.between(6, 11).mean()
    feats['evening_post_ratio'] = hours.between(18, 23).mean()

    # Burst detection
    feats['burst_count_60s'] = int((intervals < 60).sum())
    feats['burst_count_300s'] = int((intervals < 300).sum())

    # Activity spread
    feats['unique_hours'] = hours.nunique()
    feats['unique_days'] = timestamps.dt.date.nunique()
    feats['weekend_ratio'] = timestamps.dt.dayofweek.isin([5, 6]).mean()

    # Posts per day variability
    posts_per_day = timestamps.dt.date.value_counts()
    feats['posts_per_day_std'] = posts_per_day.std() if len(posts_per_day) > 1 else 0

    # Max posts in any single hour
    hour_day = timestamps.dt.floor('h')
    feats['max_posts_in_hour'] = hour_day.value_counts().max()

    # Regularity score: how "clock-like" the posting is
    # Low std of intervals relative to mean = very regular
    feats['regularity_score'] = 1.0 / (1.0 + feats['cv_interval'])

    return feats


# ============================================================
# 2c. Content Features
# ============================================================
def extract_content_features(user_posts):
    feats = {}
    texts = user_posts['text'].dropna().tolist()

    if not texts:
        return {k: 0 for k in [
            'avg_length_chars', 'avg_length_words', 'std_length_chars',
            'vocabulary_richness', 'hapax_ratio',
            'avg_hashtags', 'avg_urls', 'avg_mentions',
            'emoji_rate', 'avg_exclamation', 'avg_question',
            'avg_uppercase_ratio', 'avg_punctuation_ratio',
            'duplicate_ratio', 'near_duplicate_ratio',
            'avg_sentence_count', 'avg_word_length',
            'link_tweet_ratio', 'retweet_ratio'
        ]}

    # Length features
    char_lens = [len(t) for t in texts]
    word_lens = [len(t.split()) for t in texts]
    feats['avg_length_chars'] = np.mean(char_lens)
    feats['avg_length_words'] = np.mean(word_lens)
    feats['std_length_chars'] = np.std(char_lens)

    # Vocabulary richness
    all_words = ' '.join(texts).lower().split()
    unique_words = set(all_words)
    feats['vocabulary_richness'] = len(unique_words) / max(len(all_words), 1)
    # Hapax legomena ratio (words appearing only once)
    word_counts = Counter(all_words)
    feats['hapax_ratio'] = sum(1 for c in word_counts.values() if c == 1) / max(len(unique_words), 1)

    # Entity counts
    feats['avg_hashtags'] = np.mean([t.count('#') for t in texts])
    feats['avg_urls'] = np.mean([len(re.findall(r'https?://\S+|t\.co/\S+', t)) for t in texts])
    feats['avg_mentions'] = np.mean([t.count('@') for t in texts])
    feats['emoji_rate'] = np.mean([len(EMOJI_PATTERN.findall(t)) for t in texts])

    # Punctuation
    feats['avg_exclamation'] = np.mean([t.count('!') for t in texts])
    feats['avg_question'] = np.mean([t.count('?') for t in texts])
    feats['avg_uppercase_ratio'] = np.mean([
        sum(c.isupper() for c in t) / max(len(t), 1) for t in texts
    ])
    feats['avg_punctuation_ratio'] = np.mean([
        sum(not c.isalnum() and not c.isspace() for c in t) / max(len(t), 1)
        for t in texts
    ])

    # Duplicate / near-duplicate analysis
    unique_texts = set(texts)
    feats['duplicate_ratio'] = 1 - len(unique_texts) / max(len(texts), 1)

    near_dup = 0
    total_pairs = 0
    sample = texts[:50]
    for i in range(len(sample)):
        wi = set(sample[i].lower().split())
        for j in range(i + 1, len(sample)):
            wj = set(sample[j].lower().split())
            if wi and wj:
                jaccard = len(wi & wj) / len(wi | wj)
                if jaccard > 0.8:
                    near_dup += 1
            total_pairs += 1
    feats['near_duplicate_ratio'] = near_dup / max(total_pairs, 1)

    # Sentence & word complexity
    feats['avg_sentence_count'] = np.mean([
        len(re.split(r'[.!?]+', t)) for t in texts
    ])
    feats['avg_word_length'] = np.mean([len(w) for w in all_words]) if all_words else 0

    # Link tweets ratio
    feats['link_tweet_ratio'] = np.mean([
        1 if re.search(r'https?://|t\.co/', t) else 0 for t in texts
    ])

    # Retweet-like pattern ratio
    feats['retweet_ratio'] = np.mean([
        1 if t.strip().startswith('RT ') or t.strip().startswith('rt ') else 0
        for t in texts
    ])

    return feats


# ============================================================
# Combine all basic features for one dataset
# ============================================================
def extract_all_basic_features(posts_df, users_df):
    """Extract all Layer-1 features for every user in a dataset."""
    all_feats = []

    for _, user in users_df.iterrows():
        uid = user['id']
        user_posts = posts_df[posts_df['author_id'] == uid]

        f = {'user_id': uid}
        f.update(extract_user_profile_features(user))
        f.update(extract_behavioral_features(user_posts))
        f.update(extract_content_features(user_posts))
        all_feats.append(f)

    df = pd.DataFrame(all_feats)
    return df

print("Feature extraction functions defined.")



# Extract basic features for all datasets
basic_features = {}
for ds_id, ds in datasets.items():
    print(f"\nExtracting basic features for dataset {ds_id}...")
    feats = extract_all_basic_features(ds['posts'], ds['users'])
    feats = feats.merge(ds['users'][['id', 'is_bot']], left_on='user_id', right_on='id', how='left')
    feats.drop(columns=['id'], inplace=True)
    basic_features[ds_id] = feats
    print(f"  Shape: {feats.shape}, Bots: {(feats['is_bot']==1).sum()}, Humans: {(feats['is_bot']==0).sum()}")

print("\nBasic feature extraction complete!")
basic_features[30].head()


# ## Section 2b: Advanced Temporal Features (Enhanced)
# 
# Inspired by DARPA Bot Challenge winners, MulBot paper, and Cresci Digital DNA (2016):
# - **Time DNA Sequences**: Encode posting times as character sequences, compute self-similarity
# - **Inter-arrival Distribution Fitting**: Fit exponential distribution, measure goodness-of-fit (bots are more regular)
# - **Session Analysis**: Define activity sessions, compute session-level statistics
# - **Second-level Periodicity** (NEW v3): Second entropy, round-second ratio, interval autocorrelation, FFT peak strength


# ============================================================
# 2d. Advanced Temporal Features
# ============================================================

def encode_time_dna(timestamps, resolution='hour'):
    """
    Encode posting timestamps as a 'DNA' string sequence (Cresci et al.).
    Each character represents a time bucket. We then measure self-similarity.
    """
    if resolution == 'hour':
        return ''.join([chr(ord('A') + t.hour) for t in timestamps])
    elif resolution == 'minute_bucket':
        # 10-minute buckets (0-5 -> A-F per hour)
        return ''.join([chr(ord('A') + (t.hour * 6 + t.minute // 10)) for t in timestamps])
    return ''


def dna_self_similarity(dna_seq, k=3):
    """Compute self-similarity of a DNA sequence using k-mer overlap."""
    if len(dna_seq) < k + 1:
        return 0.0
    kmers = [dna_seq[i:i+k] for i in range(len(dna_seq) - k + 1)]
    unique_kmers = set(kmers)
    return 1.0 - (len(unique_kmers) / max(len(kmers), 1))


def extract_advanced_temporal_features(user_posts):
    """Extract advanced temporal features per user."""
    feats = {}
    n = len(user_posts)

    default_feats = {
        'time_dna_self_sim_hour': 0, 'time_dna_self_sim_minute': 0,
        'time_dna_unique_3gram_ratio': 0,
        'iat_exponential_ks_stat': 0, 'iat_exponential_ks_pvalue': 1.0,
        'iat_gini_coefficient': 0,
        'iat_benford_deviation': 0,
        'session_count': 0, 'avg_session_length': 0, 'max_session_length': 0,
        'avg_inter_session_gap': 0, 'session_regularity': 0,
        'posting_acceleration': 0,
        'longest_active_streak_hours': 0,
        'minute_entropy': 0,
        'day_of_week_entropy': 0,
        # NEW: Second-level periodicity features (Cresci Digital DNA, 2016)
        'second_entropy': 0,
        'second_mode_frequency': 0,
        'round_second_ratio': 0,
        'interval_autocorr_lag1': 0,
        'interval_fft_peak_strength': 0,
    }

    if n < 3:
        return default_feats

    try:
        timestamps = pd.to_datetime(user_posts['created_at']).sort_values().reset_index(drop=True)
    except:
        return default_feats

    # --- Time DNA Sequences ---
    dna_hour = encode_time_dna(timestamps, 'hour')
    dna_minute = encode_time_dna(timestamps, 'minute_bucket')
    feats['time_dna_self_sim_hour'] = dna_self_similarity(dna_hour, k=3)
    feats['time_dna_self_sim_minute'] = dna_self_similarity(dna_minute, k=3)

    # 3-gram uniqueness ratio for time DNA
    if len(dna_hour) >= 4:
        kmers = [dna_hour[i:i+3] for i in range(len(dna_hour) - 2)]
        feats['time_dna_unique_3gram_ratio'] = len(set(kmers)) / max(len(kmers), 1)
    else:
        feats['time_dna_unique_3gram_ratio'] = 0

    # --- Inter-arrival Time Distribution Fitting ---
    intervals = timestamps.diff().dropna().dt.total_seconds().values
    intervals = intervals[intervals > 0]  # Remove zero intervals

    if len(intervals) >= 5:
        # KS test against exponential distribution (human posting tends to be heavy-tailed)
        try:
            loc, scale = stats.expon.fit(intervals, floc=0)
            ks_stat, ks_pvalue = stats.kstest(intervals, 'expon', args=(0, scale))
            feats['iat_exponential_ks_stat'] = float(ks_stat)
            feats['iat_exponential_ks_pvalue'] = float(ks_pvalue)
        except:
            feats['iat_exponential_ks_stat'] = 0
            feats['iat_exponential_ks_pvalue'] = 1.0

        # Gini coefficient of intervals (measures inequality; bots tend to be more equal = lower Gini)
        sorted_intervals = np.sort(intervals)
        n_int = len(sorted_intervals)
        index = np.arange(1, n_int + 1)
        feats['iat_gini_coefficient'] = float(
            (2 * np.sum(index * sorted_intervals) / (n_int * np.sum(sorted_intervals))) - (n_int + 1) / n_int
        ) if np.sum(sorted_intervals) > 0 else 0

        # Benford's law deviation on leading digit of intervals
        leading_digits = [int(str(abs(int(x)))[0]) for x in intervals if x >= 1]
        if leading_digits:
            digit_counts = Counter(leading_digits)
            total = sum(digit_counts.values())
            benford_expected = {d: np.log10(1 + 1/d) for d in range(1, 10)}
            deviation = sum(
                abs(digit_counts.get(d, 0) / total - benford_expected[d])
                for d in range(1, 10)
            )
            feats['iat_benford_deviation'] = float(deviation)
        else:
            feats['iat_benford_deviation'] = 0
    else:
        feats['iat_exponential_ks_stat'] = 0
        feats['iat_exponential_ks_pvalue'] = 1.0
        feats['iat_gini_coefficient'] = 0
        feats['iat_benford_deviation'] = 0

    # --- Session Analysis ---
    SESSION_GAP_SECONDS = 1800  # 30-minute gap defines a new session
    if len(intervals) >= 1:
        session_breaks = np.where(intervals > SESSION_GAP_SECONDS)[0]
        session_count = len(session_breaks) + 1
        feats['session_count'] = session_count

        # Compute session lengths (number of posts per session)
        session_starts = np.concatenate([[0], session_breaks + 1])
        session_ends = np.concatenate([session_breaks + 1, [len(timestamps)]])
        session_lengths = session_ends - session_starts

        feats['avg_session_length'] = float(np.mean(session_lengths))
        feats['max_session_length'] = float(np.max(session_lengths))

        # Inter-session gaps
        if session_count > 1:
            inter_session_gaps = intervals[session_breaks]
            feats['avg_inter_session_gap'] = float(np.mean(inter_session_gaps))
            feats['session_regularity'] = float(
                np.std(inter_session_gaps) / max(np.mean(inter_session_gaps), 0.01)
            )
        else:
            feats['avg_inter_session_gap'] = 0
            feats['session_regularity'] = 0
    else:
        feats['session_count'] = 1
        feats['avg_session_length'] = n
        feats['max_session_length'] = n
        feats['avg_inter_session_gap'] = 0
        feats['session_regularity'] = 0

    # --- Posting Acceleration ---
    # Compare posting frequency in first half vs second half of time span
    mid_idx = len(timestamps) // 2
    if mid_idx > 0 and mid_idx < len(timestamps) - 1:
        first_half_span = (timestamps.iloc[mid_idx] - timestamps.iloc[0]).total_seconds()
        second_half_span = (timestamps.iloc[-1] - timestamps.iloc[mid_idx]).total_seconds()
        freq_first = mid_idx / max(first_half_span, 1)
        freq_second = (len(timestamps) - mid_idx) / max(second_half_span, 1)
        feats['posting_acceleration'] = freq_second - freq_first
    else:
        feats['posting_acceleration'] = 0

    # --- Longest Active Streak ---
    # How many hours was the user continuously active (at least 1 post per hour)
    hour_bins = timestamps.dt.floor('h')
    active_hours = sorted(hour_bins.unique())
    if len(active_hours) > 1:
        diffs = [(active_hours[i+1] - active_hours[i]).total_seconds() / 3600
                 for i in range(len(active_hours) - 1)]
        longest_streak = 1
        current_streak = 1
        for d in diffs:
            if d <= 1.0:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 1
        feats['longest_active_streak_hours'] = longest_streak
    else:
        feats['longest_active_streak_hours'] = 1

    # --- Minute-level entropy ---
    minutes = timestamps.dt.minute
    minute_counts = minutes.value_counts(normalize=True)
    feats['minute_entropy'] = float(stats.entropy(minute_counts))

    # --- Day of week entropy ---
    dow = timestamps.dt.dayofweek
    dow_counts = dow.value_counts(normalize=True)
    feats['day_of_week_entropy'] = float(stats.entropy(dow_counts))

    # === NEW: Second-level Periodicity Features ===
    # Bots often post at exact time points (e.g., :00, :30 seconds) due to scheduled scripts.
    # Reference: Cresci et al. "DNA-inspired online behavioral modeling" (2016)

    # --- Second-level entropy ---
    seconds = timestamps.dt.second
    second_counts = seconds.value_counts(normalize=True)
    feats['second_entropy'] = float(stats.entropy(second_counts))

    # --- Mode frequency of seconds (most common second value) ---
    if len(seconds) > 0:
        mode_count = seconds.value_counts().iloc[0]
        feats['second_mode_frequency'] = mode_count / len(seconds)
    else:
        feats['second_mode_frequency'] = 0

    # --- Round second ratio (posts at :00 or :30 seconds) ---
    round_secs = seconds.isin([0, 30]).sum()
    feats['round_second_ratio'] = round_secs / max(len(seconds), 1)

    # --- Inter-arrival time autocorrelation (lag-1) ---
    # High autocorrelation = very regular intervals = bot-like
    if len(intervals) >= 10:
        autocorr = np.corrcoef(intervals[:-1], intervals[1:])[0, 1]
        feats['interval_autocorr_lag1'] = float(autocorr) if not np.isnan(autocorr) else 0
    else:
        feats['interval_autocorr_lag1'] = 0

    # --- FFT peak strength for periodicity detection ---
    # Strong spectral peaks = periodic posting pattern = bot-like
    if len(intervals) >= 16:
        fft_vals = np.abs(np.fft.rfft(intervals - np.mean(intervals)))
        if len(fft_vals) > 1:
            fft_vals = fft_vals[1:]  # Remove DC component
            feats['interval_fft_peak_strength'] = float(np.max(fft_vals) / (np.mean(fft_vals) + 1e-8))
        else:
            feats['interval_fft_peak_strength'] = 0
    else:
        feats['interval_fft_peak_strength'] = 0

    return feats


# Apply advanced temporal features to all datasets
advanced_temporal_features = {}
for ds_id, ds in datasets.items():
    print(f"\nExtracting advanced temporal features for dataset {ds_id}...")
    results = []
    for _, user in ds['users'].iterrows():
        uid = user['id']
        user_posts = ds['posts'][ds['posts']['author_id'] == uid]
        f = {'user_id': uid}
        f.update(extract_advanced_temporal_features(user_posts))
        results.append(f)
    advanced_temporal_features[ds_id] = pd.DataFrame(results)
    print(f"  Shape: {advanced_temporal_features[ds_id].shape}")

print("\nAdvanced temporal feature extraction complete!")


# ## Section 2c: Text Stylometry Features (NEW)
# 
# Inspired by TwiBot-22 baselines (Lee's compression ratio, Kantepe's text entropy):
# - **N-gram Repetition**: 2-gram and 3-gram repetition rates across tweets
# - **Pairwise Jaccard Similarity**: Average Jaccard between tweet pairs (bot tweets are template-like)
# - **Zipf's Law Deviation**: Human text follows Zipf's law; bots may deviate
# - **Compression Ratio**: How compressible the user's combined text is (bots = more compressible)
# - **Text Entropy**: Shannon entropy of character distribution
# - **Punctuation Pattern Features**: Bot punctuation usage is often more regular


# ============================================================
# 2e. Text Stylometry Features
# ============================================================

def compute_ngram_repetition(texts, n=2):
    """Compute n-gram repetition rate across all tweets."""
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / max(len(all_ngrams), 1)


def compute_zipf_deviation(texts):
    """
    Measure deviation from Zipf's law.
    Zipf's law: frequency of word rank r is proportional to 1/r.
    Returns the mean squared error between observed and expected Zipf distribution.
    """
    all_words = ' '.join(texts).lower().split()
    if len(all_words) < 10:
        return 0.0

    word_counts = Counter(all_words)
    frequencies = sorted(word_counts.values(), reverse=True)
    ranks = np.arange(1, len(frequencies) + 1)
    freq_arr = np.array(frequencies, dtype=float)
    freq_arr /= freq_arr[0]  # Normalize to first rank

    # Expected Zipf: f(r) = 1/r
    expected = 1.0 / ranks
    # Use only top 50 to avoid noise
    limit = min(50, len(freq_arr))
    mse = float(np.mean((freq_arr[:limit] - expected[:limit]) ** 2))
    return mse


def compute_text_entropy(text):
    """Compute Shannon entropy of character distribution."""
    if not text:
        return 0.0
    char_counts = Counter(text.lower())
    total = sum(char_counts.values())
    probs = [c / total for c in char_counts.values()]
    return float(stats.entropy(probs))


def compute_compression_ratio(texts):
    """
    Compute compression ratio of combined text (Lee et al.).
    More compressible = more repetitive = more bot-like.
    """
    combined = ' '.join(texts).encode('utf-8')
    if len(combined) < 10:
        return 1.0
    compressed = zlib.compress(combined)
    return len(combined) / max(len(compressed), 1)


def extract_stylometry_features(user_posts):
    """Extract text stylometry features per user."""
    feats = {}
    texts = user_posts['text'].dropna().tolist()

    default_feats = {
        'ngram2_repetition': 0, 'ngram3_repetition': 0,
        'pairwise_jaccard_mean': 0, 'pairwise_jaccard_std': 0,
        'pairwise_jaccard_max': 0,
        'zipf_deviation': 0,
        'compression_ratio': 1.0,
        'text_char_entropy': 0,
        'punctuation_pattern_std': 0,
        'sentence_length_cv': 0,
        'avg_word_length_std': 0,
        'unique_first_words_ratio': 0,
        'url_pattern_regularity': 0,
        'mention_diversity': 0,
        'hashtag_diversity': 0,
    }

    if len(texts) < 3:
        return default_feats

    # --- N-gram Repetition ---
    feats['ngram2_repetition'] = compute_ngram_repetition(texts, n=2)
    feats['ngram3_repetition'] = compute_ngram_repetition(texts, n=3)

    # --- Pairwise Jaccard Similarity ---
    sample = texts[:50]
    jaccard_scores = []
    for i in range(len(sample)):
        wi = set(sample[i].lower().split())
        for j in range(i + 1, min(i + 10, len(sample))):  # Limit pairs for speed
            wj = set(sample[j].lower().split())
            if wi or wj:
                jaccard_scores.append(len(wi & wj) / max(len(wi | wj), 1))
    if jaccard_scores:
        feats['pairwise_jaccard_mean'] = float(np.mean(jaccard_scores))
        feats['pairwise_jaccard_std'] = float(np.std(jaccard_scores))
        feats['pairwise_jaccard_max'] = float(np.max(jaccard_scores))
    else:
        feats['pairwise_jaccard_mean'] = 0
        feats['pairwise_jaccard_std'] = 0
        feats['pairwise_jaccard_max'] = 0

    # --- Zipf's Law Deviation ---
    feats['zipf_deviation'] = compute_zipf_deviation(texts)

    # --- Compression Ratio ---
    feats['compression_ratio'] = compute_compression_ratio(texts)

    # --- Character Entropy ---
    combined_text = ' '.join(texts)
    feats['text_char_entropy'] = compute_text_entropy(combined_text)

    # --- Punctuation Pattern Regularity ---
    # Std of punctuation count per tweet (regular = bot-like)
    punct_counts = [sum(1 for c in t if not c.isalnum() and not c.isspace()) for t in texts]
    feats['punctuation_pattern_std'] = float(np.std(punct_counts)) if punct_counts else 0

    # --- Sentence Length Coefficient of Variation ---
    word_lengths = [len(t.split()) for t in texts]
    mean_wl = np.mean(word_lengths)
    feats['sentence_length_cv'] = float(np.std(word_lengths) / max(mean_wl, 0.01))

    # --- Word Length Variability Across Tweets ---
    avg_word_lens = [np.mean([len(w) for w in t.split()]) if t.split() else 0 for t in texts]
    feats['avg_word_length_std'] = float(np.std(avg_word_lens))

    # --- Unique First Words Ratio (template detection) ---
    first_words = [t.split()[0].lower() if t.split() else '' for t in texts]
    feats['unique_first_words_ratio'] = len(set(first_words)) / max(len(first_words), 1)

    # --- URL/Mention/Hashtag Diversity ---
    all_urls = [url for t in texts for url in re.findall(r'https?://\S+|t\.co/\S+', t)]
    feats['url_pattern_regularity'] = 1 - (len(set(all_urls)) / max(len(all_urls), 1)) if all_urls else 0

    all_mentions = [m for t in texts for m in re.findall(r'@\w+', t)]
    feats['mention_diversity'] = len(set(all_mentions)) / max(len(all_mentions), 1) if all_mentions else 0

    all_hashtags = [h for t in texts for h in re.findall(r'#\w+', t)]
    feats['hashtag_diversity'] = len(set(all_hashtags)) / max(len(all_hashtags), 1) if all_hashtags else 0

    return feats


# Apply stylometry features to all datasets
stylometry_features = {}
for ds_id, ds in datasets.items():
    print(f"\nExtracting stylometry features for dataset {ds_id}...")
    results = []
    for _, user in ds['users'].iterrows():
        uid = user['id']
        user_posts = ds['posts'][ds['posts']['author_id'] == uid]
        f = {'user_id': uid}
        f.update(extract_stylometry_features(user_posts))
        results.append(f)
    stylometry_features[ds_id] = pd.DataFrame(results)
    print(f"  Shape: {stylometry_features[ds_id].shape}")

print("\nStylometry feature extraction complete!")


# ## Section 2d: Mention Network Features (NEW)
# 
# Inspired by BIC (Wei & Nguyen, ACL 2023) and RoGBot (2025):
# - **Mention Entropy**: Distribution of mentioned users (bots often mention same targets)
# - **In-dataset Mention Ratio**: Fraction of mentions targeting users in the dataset (coordinated bots mention each other)
# - **Mention Concentration**: How concentrated mentions are on top targets
# - **Mention Reciprocity**: Whether mentioned users mention back (bots in a group often reciprocate)
# - **Is-Mentioned Score**: How many other users mention this user


# ============================================================
# 2f. Mention Network Features (NEW)
# ============================================================
# Reference: BIC (ACL 2023) - interaction patterns without follower data
# Reference: RoGBot (2025) - relationship-oblivious graph-based detection

def build_mention_graph(posts_df, users_df):
    """
    Build a mention graph from tweets.
    Returns:
        mention_by_user: dict {user_id: Counter of mentioned usernames}
        username_to_id: dict {username: user_id} for users in dataset
        mentioned_by: dict {user_id: set of user_ids who mentioned them}
    """
    # Build username -> user_id mapping for dataset users
    username_to_id = {}
    id_to_username = {}
    for _, user in users_df.iterrows():
        uname = str(user.get('username', '')).lower()
        if uname:
            username_to_id[uname] = user['id']
            id_to_username[user['id']] = uname

    # Extract mentions per user
    mention_pattern = re.compile(r'@(\w+)')
    mention_by_user = {}  # {user_id: Counter of mentioned usernames}

    for _, row in posts_df.iterrows():
        author_id = row['author_id']
        text = str(row.get('text', ''))
        mentions = [m.lower() for m in mention_pattern.findall(text)]
        if author_id not in mention_by_user:
            mention_by_user[author_id] = Counter()
        mention_by_user[author_id].update(mentions)

    # Build reverse map: who mentions whom (by user_id)
    mentioned_by = {uid: set() for uid in users_df['id']}
    for uid, mention_counts in mention_by_user.items():
        for mentioned_uname in mention_counts:
            if mentioned_uname in username_to_id:
                mentioned_uid = username_to_id[mentioned_uname]
                if mentioned_uid != uid:  # exclude self
                    mentioned_by[mentioned_uid].add(uid)

    return mention_by_user, username_to_id, mentioned_by


def extract_mention_features(user_id, mention_by_user, username_to_id, mentioned_by, all_user_ids):
    """Extract mention network features for a single user."""
    feats = {}
    mentions = mention_by_user.get(user_id, Counter())
    total_mentions = sum(mentions.values())

    if total_mentions == 0:
        return {
            'mention_total_count': 0,
            'mention_unique_count': 0,
            'mention_unique_ratio': 0,
            'mention_entropy': 0,
            'mention_concentration_top3': 0,
            'mention_in_dataset_ratio': 0,
            'mention_self_ratio': 0,
            'is_mentioned_count': 0,
            'is_mentioned_ratio': 0,
            'mention_reciprocity': 0,
        }

    # Basic counts
    feats['mention_total_count'] = total_mentions
    feats['mention_unique_count'] = len(mentions)
    feats['mention_unique_ratio'] = len(mentions) / max(total_mentions, 1)

    # Mention entropy (diversity of mentioned users)
    mention_probs = np.array(list(mentions.values()), dtype=float)
    mention_probs /= mention_probs.sum()
    feats['mention_entropy'] = float(stats.entropy(mention_probs))

    # Concentration: top 3 mentioned users' share of all mentions
    top3 = sum(c for _, c in mentions.most_common(3))
    feats['mention_concentration_top3'] = top3 / max(total_mentions, 1)

    # In-dataset mention ratio (mentions targeting users in the dataset)
    in_dataset_mentions = sum(
        c for uname, c in mentions.items() if uname in username_to_id
    )
    feats['mention_in_dataset_ratio'] = in_dataset_mentions / max(total_mentions, 1)

    # Self-mention ratio
    own_username = None
    for uname, uid in username_to_id.items():
        if uid == user_id:
            own_username = uname
            break
    self_mentions = mentions.get(own_username, 0) if own_username else 0
    feats['mention_self_ratio'] = self_mentions / max(total_mentions, 1)

    # Is-mentioned score: how many other users mention this user
    is_mentioned_set = mentioned_by.get(user_id, set())
    feats['is_mentioned_count'] = len(is_mentioned_set)
    feats['is_mentioned_ratio'] = len(is_mentioned_set) / max(len(all_user_ids) - 1, 1)

    # Reciprocity: of the users this user mentions (in-dataset), how many mention back
    mentioned_in_dataset = set()
    for uname in mentions:
        if uname in username_to_id:
            mentioned_in_dataset.add(username_to_id[uname])
    mentioned_in_dataset.discard(user_id)  # remove self

    if mentioned_in_dataset:
        reciprocal = sum(
            1 for uid in mentioned_in_dataset
            if user_id in mentioned_by.get(uid, set())
        )
        feats['mention_reciprocity'] = reciprocal / len(mentioned_in_dataset)
    else:
        feats['mention_reciprocity'] = 0

    return feats


# Apply mention features to all datasets
mention_features = {}
for ds_id, ds in datasets.items():
    print(f"\nExtracting mention network features for dataset {ds_id}...")
    mention_by_user, username_to_id, mentioned_by = build_mention_graph(
        ds['posts'], ds['users'])
    all_user_ids = set(ds['users']['id'])

    results = []
    for _, user in ds['users'].iterrows():
        uid = user['id']
        f = {'user_id': uid}
        f.update(extract_mention_features(
            uid, mention_by_user, username_to_id, mentioned_by, all_user_ids))
        results.append(f)
    mention_features[ds_id] = pd.DataFrame(results)
    print(f"  Shape: {mention_features[ds_id].shape}")

print("\nMention network feature extraction complete!")


# ## Section 3: Deep NLP Features (Layer 2)
# 
# - **Sentence Transformer**: Compute per-user tweet embedding similarity (bot tweets may be too uniform or too random)
# - **Multilingual Perplexity** (FIXED): Language-specific models - GPT-2 for English, BLOOM-560m for French
#   - Previous version used English GPT-2 for French text, producing meaningless scores


# ============================================================
# 3a. Sentence Transformer Embedding Features
# ============================================================
# Using multilingual model so it works for both English and French
EMBED_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def extract_embedding_features(posts_df, users_df, model=None):
    """Compute per-user tweet embedding similarity features."""
    if model is None:
        model = SentenceTransformer(EMBED_MODEL_NAME)

    results = []
    mean_embeddings = {}  # Store for cross-user analysis later

    for _, user in users_df.iterrows():
        uid = user['id']
        user_posts = posts_df[posts_df['author_id'] == uid]
        texts = user_posts['text'].dropna().tolist()

        feats = {'user_id': uid}

        if len(texts) < 2:
            feats.update({
                'emb_avg_sim': 0, 'emb_std_sim': 0,
                'emb_min_sim': 0, 'emb_max_sim': 0,
                'emb_median_sim': 0
            })
            results.append(feats)
            continue

        # Sample up to 50 tweets for performance
        sample = texts[:50]
        embeddings = model.encode(sample, show_progress_bar=False, batch_size=32)

        # Store mean embedding for cross-user analysis
        mean_embeddings[uid] = np.mean(embeddings, axis=0)

        # Pairwise cosine similarities (upper triangle)
        sim_matrix = cosine_similarity(embeddings)
        upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

        feats['emb_avg_sim'] = float(np.mean(upper_tri))
        feats['emb_std_sim'] = float(np.std(upper_tri))
        feats['emb_min_sim'] = float(np.min(upper_tri))
        feats['emb_max_sim'] = float(np.max(upper_tri))
        feats['emb_median_sim'] = float(np.median(upper_tri))

        results.append(feats)

    return pd.DataFrame(results), mean_embeddings


# ============================================================
# 3b. Multilingual Perplexity Features (FIXED for French)
# ============================================================
# GPT-2 is English-only and produces meaningless perplexity for French.
# We use language-specific models:
#   - English: 'gpt2' (standard GPT-2)
#   - French: 'bigscience/bloom-560m' (multilingual, strong French support)
# Reference: BLOOM (BigScience, 2022), mGPT (Shliazhko et al., 2024)

PPL_MODELS = {
    'en': 'gpt2',
    'fr': 'bigscience/bloom-560m',
}

def load_perplexity_model(lang='en'):
    """Load language-appropriate causal LM for perplexity computation.
    English: GPT-2, French: BLOOM-560m (multilingual with strong French support).
    Uses float16 on GPU to save VRAM (especially important for BLOOM-560m).
    """
    model_name = PPL_MODELS.get(lang, 'gpt2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Loading perplexity model for '{lang}': {model_name} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use float16 on GPU to reduce VRAM usage (~50% savings)
    if device == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    # Ensure pad token is set (BLOOM needs this)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def compute_perplexity(text, model, tokenizer, device, max_length=512):
    """Compute perplexity of a text using a causal language model."""
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                          max_length=max_length, padding=False).to(device)
        if inputs['input_ids'].shape[1] < 2:
            return None
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        return float(torch.exp(outputs.loss))
    except:
        return None


def extract_perplexity_features(posts_df, users_df, ppl_model, ppl_tokenizer, ppl_device):
    """Compute per-user perplexity statistics."""
    results = []

    for idx, (_, user) in enumerate(users_df.iterrows()):
        uid = user['id']
        user_posts = posts_df[posts_df['author_id'] == uid]
        texts = user_posts['text'].dropna().tolist()

        feats = {'user_id': uid}

        if not texts:
            feats.update({
                'ppl_mean': 0, 'ppl_std': 0, 'ppl_min': 0,
                'ppl_max': 0, 'ppl_median': 0, 'ppl_skew': 0,
                'ppl_low_ratio': 0
            })
            results.append(feats)
            continue

        # Sample up to 20 tweets for performance
        sample = texts[:20]
        perplexities = [compute_perplexity(t, ppl_model, ppl_tokenizer, ppl_device)
                       for t in sample]
        perplexities = [p for p in perplexities if p is not None and p < 10000]

        if perplexities:
            feats['ppl_mean'] = np.mean(perplexities)
            feats['ppl_std'] = np.std(perplexities)
            feats['ppl_min'] = np.min(perplexities)
            feats['ppl_max'] = np.max(perplexities)
            feats['ppl_median'] = np.median(perplexities)
            feats['ppl_skew'] = float(stats.skew(perplexities)) if len(perplexities) >= 3 else 0
            # Ratio of tweets with unusually low perplexity (< 50 = very "fluent")
            feats['ppl_low_ratio'] = np.mean([1 if p < 50 else 0 for p in perplexities])
        else:
            feats.update({
                'ppl_mean': 0, 'ppl_std': 0, 'ppl_min': 0,
                'ppl_max': 0, 'ppl_median': 0, 'ppl_skew': 0,
                'ppl_low_ratio': 0
            })

        results.append(feats)

        if (idx + 1) % 50 == 0:
            print(f"  Perplexity: {idx+1}/{len(users_df)} users processed")

    return pd.DataFrame(results)

print("NLP feature functions defined.")



# Load models once
print("Loading Sentence Transformer...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Load per-language perplexity models (FIXED: no longer using English GPT-2 for French)
ppl_models = {}
for lang_code in ['en', 'fr']:
    print(f"\nLoading perplexity model for '{lang_code}'...")
    ppl_m, ppl_t, ppl_d = load_perplexity_model(lang=lang_code)
    ppl_models[lang_code] = {'model': ppl_m, 'tokenizer': ppl_t, 'device': ppl_d}
print(f"\nAll models loaded. Device: {ppl_models['en']['device']}")

# Extract NLP features for all datasets
nlp_features = {}
all_mean_embeddings = {}

for ds_id, ds in datasets.items():
    ds_lang = ds['meta']['lang']
    print(f"\n{'='*50}")
    print(f"Dataset {ds_id} ({ds_lang})")
    print(f"{'='*50}")

    # Embedding features
    print("Computing sentence embeddings...")
    emb_feats, mean_embs = extract_embedding_features(
        ds['posts'], ds['users'], model=embed_model)

    # Perplexity features (using language-appropriate model)
    ppl_info = ppl_models.get(ds_lang, ppl_models['en'])
    print(f"Computing perplexity scores (model: {PPL_MODELS.get(ds_lang, 'gpt2')})...")
    ppl_feats = extract_perplexity_features(
        ds['posts'], ds['users'],
        ppl_info['model'], ppl_info['tokenizer'], ppl_info['device'])

    # Merge
    nlp_feat = emb_feats.merge(ppl_feats, on='user_id', how='outer')
    nlp_features[ds_id] = nlp_feat
    all_mean_embeddings[ds_id] = mean_embs

    print(f"  NLP features shape: {nlp_feat.shape}")

print("\nNLP feature extraction complete!")


# ## Section 3c: Topic Coherence & Diversity Features (NEW)
# 
# Inspired by BIC (ACL 2023) semantic consistency analysis:
# - **Sequential Coherence**: Average cosine similarity between consecutive tweet embeddings (template bots = very high, stolen-tweet bots = very low)
# - **Topic Diversity Score**: Mean pairwise distance of embeddings (monotopic bots have low diversity)
# - **Topic Cluster Count**: Number of distinct topic clusters via K-Means on embeddings
# - **Topic Switch Rate**: Fraction of consecutive tweet pairs with low similarity (topic "jumps")


# ============================================================
# 3c. Topic Coherence & Diversity Features (NEW)
# ============================================================
# Reference: BIC (Wei & Nguyen, ACL 2023) - semantic consistency across timelines
# Reference: "Bots Don't Sit Still" (2025) - content feature evolution

def extract_topic_coherence_features(posts_df, users_df, embed_model):
    """
    Extract topic coherence and diversity features per user using embeddings.
    These features capture whether a user's tweets are topically consistent
    (template bots), diverse (normal humans), or inconsistent (stolen-tweet bots).
    """
    results = []

    for _, user in users_df.iterrows():
        uid = user['id']
        user_posts = posts_df[posts_df['author_id'] == uid]
        texts = user_posts['text'].dropna().tolist()

        feats = {'user_id': uid}

        default_feats = {
            'sequential_coherence_mean': 0,
            'sequential_coherence_std': 0,
            'sequential_coherence_min': 0,
            'topic_diversity_score': 0,
            'topic_cluster_count': 0,
            'topic_dominant_cluster_ratio': 0,
            'topic_switch_rate': 0,
            'topic_embedding_spread': 0,
        }

        if len(texts) < 4:
            feats.update(default_feats)
            results.append(feats)
            continue

        # Encode tweets (sample up to 50 for speed)
        sample = texts[:50]
        embeddings = embed_model.encode(sample, show_progress_bar=False, batch_size=32)

        # --- Sequential Coherence ---
        # Cosine similarity between consecutive tweets (in chronological order)
        sequential_sims = []
        for i in range(len(embeddings) - 1):
            sim = 1 - cosine(embeddings[i], embeddings[i+1])
            if not np.isnan(sim):
                sequential_sims.append(sim)

        if sequential_sims:
            feats['sequential_coherence_mean'] = float(np.mean(sequential_sims))
            feats['sequential_coherence_std'] = float(np.std(sequential_sims))
            feats['sequential_coherence_min'] = float(np.min(sequential_sims))
        else:
            feats['sequential_coherence_mean'] = 0
            feats['sequential_coherence_std'] = 0
            feats['sequential_coherence_min'] = 0

        # --- Topic Diversity Score ---
        # Mean pairwise cosine distance (higher = more diverse topics)
        sim_matrix = cosine_similarity(embeddings)
        upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        feats['topic_diversity_score'] = float(1.0 - np.mean(upper_tri))

        # --- Embedding Spread ---
        # Standard deviation of embedding vectors from their centroid
        centroid = np.mean(embeddings, axis=0)
        distances = [np.linalg.norm(e - centroid) for e in embeddings]
        feats['topic_embedding_spread'] = float(np.std(distances))

        # --- Topic Cluster Count (K-Means) ---
        # Estimate number of distinct topics using silhouette-like heuristic
        max_k = min(5, len(embeddings) - 1)
        if max_k >= 2:
            best_k = 1
            best_inertia_drop = 0
            prev_inertia = None
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=50)
                kmeans.fit(embeddings)
                if prev_inertia is not None:
                    drop = prev_inertia - kmeans.inertia_
                    if drop > best_inertia_drop:
                        best_inertia_drop = drop
                        best_k = k
                prev_inertia = kmeans.inertia_

            feats['topic_cluster_count'] = best_k

            # Dominant cluster ratio
            kmeans_final = KMeans(n_clusters=max(best_k, 2), n_init=3, random_state=42, max_iter=50)
            labels = kmeans_final.fit_predict(embeddings)
            label_counts = Counter(labels)
            dominant_count = label_counts.most_common(1)[0][1]
            feats['topic_dominant_cluster_ratio'] = dominant_count / len(embeddings)
        else:
            feats['topic_cluster_count'] = 1
            feats['topic_dominant_cluster_ratio'] = 1.0

        # --- Topic Switch Rate ---
        # Fraction of consecutive pairs with similarity below threshold
        switch_threshold = 0.3  # Low similarity = topic switch
        if sequential_sims:
            switches = sum(1 for s in sequential_sims if s < switch_threshold)
            feats['topic_switch_rate'] = switches / len(sequential_sims)
        else:
            feats['topic_switch_rate'] = 0

        results.append(feats)

    return pd.DataFrame(results)


# Apply topic coherence features to all datasets
topic_coherence_features = {}
for ds_id, ds in datasets.items():
    print(f"\nExtracting topic coherence features for dataset {ds_id}...")
    topic_feats = extract_topic_coherence_features(
        ds['posts'], ds['users'], embed_model)
    topic_coherence_features[ds_id] = topic_feats
    print(f"  Shape: {topic_feats.shape}")

print("\nTopic coherence feature extraction complete!")


# ## Section 3d: Sentiment Analysis Features (NEW)
# 
# Inspired by "Bots Don't Sit Still" (2025) longitudinal study:
# - **Sentiment Polarity**: Mean and std of sentiment scores (bots may have unnatural sentiment patterns)
# - **Sentiment Extreme Ratio**: Fraction of tweets with extreme positive/negative sentiment
# - **Sentiment Monotony**: How constant the sentiment is across tweets (bots often have uniform sentiment)
# 
# Uses `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` which supports both English and French.


# ============================================================
# 3d. Sentiment Analysis Features (NEW)
# ============================================================
# Reference: "Bots Don't Sit Still" (2025) - sentiment is a key feature
# that distinguishes bots from humans and evolves over time.
# Using multilingual model for both English and French support.

from transformers import pipeline as hf_pipeline

SENTIMENT_MODEL = 'cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual'

def load_sentiment_model():
    """Load multilingual sentiment analysis pipeline."""
    device_idx = 0 if torch.cuda.is_available() else -1
    sentiment_pipe = hf_pipeline(
        'sentiment-analysis',
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        device=device_idx,
        truncation=True,
        max_length=512
    )
    return sentiment_pipe


def extract_sentiment_features(posts_df, users_df, sentiment_pipe):
    """Extract sentiment features per user using multilingual sentiment model."""
    results = []

    # Label-to-score mapping: negative=-1, neutral=0, positive=1
    label_map = {'negative': -1.0, 'neutral': 0.0, 'positive': 1.0}

    for idx, (_, user) in enumerate(users_df.iterrows()):
        uid = user['id']
        user_posts = posts_df[posts_df['author_id'] == uid]
        texts = user_posts['text'].dropna().tolist()

        feats = {'user_id': uid}

        default_feats = {
            'sentiment_mean': 0,
            'sentiment_std': 0,
            'sentiment_positive_ratio': 0,
            'sentiment_negative_ratio': 0,
            'sentiment_neutral_ratio': 0,
            'sentiment_extreme_ratio': 0,
            'sentiment_monotony': 0,
            'sentiment_confidence_mean': 0,
            'sentiment_confidence_std': 0,
        }

        if len(texts) < 3:
            feats.update(default_feats)
            results.append(feats)
            continue

        # Sample up to 30 tweets for speed
        sample = texts[:30]

        try:
            predictions = sentiment_pipe(sample, batch_size=16)
        except:
            feats.update(default_feats)
            results.append(feats)
            continue

        scores = []
        labels = []
        confidences = []
        for pred in predictions:
            label = pred['label'].lower()
            conf = pred['score']
            labels.append(label)
            confidences.append(conf)
            scores.append(label_map.get(label, 0.0))

        scores = np.array(scores)
        confidences = np.array(confidences)

        # Mean and std of sentiment polarity
        feats['sentiment_mean'] = float(np.mean(scores))
        feats['sentiment_std'] = float(np.std(scores))

        # Sentiment distribution ratios
        label_counter = Counter(labels)
        n_preds = len(labels)
        feats['sentiment_positive_ratio'] = label_counter.get('positive', 0) / n_preds
        feats['sentiment_negative_ratio'] = label_counter.get('negative', 0) / n_preds
        feats['sentiment_neutral_ratio'] = label_counter.get('neutral', 0) / n_preds

        # Extreme ratio: strong positive or strong negative (non-neutral)
        feats['sentiment_extreme_ratio'] = 1.0 - feats['sentiment_neutral_ratio']

        # Monotony: low std = very uniform sentiment = potentially bot-like
        # Normalized to 0-1 where 1 = perfectly monotone
        feats['sentiment_monotony'] = 1.0 / (1.0 + feats['sentiment_std'])

        # Confidence stats
        feats['sentiment_confidence_mean'] = float(np.mean(confidences))
        feats['sentiment_confidence_std'] = float(np.std(confidences))

        results.append(feats)

        if (idx + 1) % 50 == 0:
            print(f"  Sentiment: {idx+1}/{len(users_df)} users processed")

    return pd.DataFrame(results)


# Load sentiment model
print("Loading multilingual sentiment model...")
sentiment_pipe = load_sentiment_model()
print(f"Sentiment model loaded: {SENTIMENT_MODEL}")

# Apply sentiment features to all datasets
sentiment_features = {}
for ds_id, ds in datasets.items():
    print(f"\nExtracting sentiment features for dataset {ds_id} ({ds['meta']['lang']})...")
    sent_feats = extract_sentiment_features(ds['posts'], ds['users'], sentiment_pipe)
    sentiment_features[ds_id] = sent_feats
    print(f"  Shape: {sent_feats.shape}")

print("\nSentiment feature extraction complete!")


# ## Section 4: Cross-User Features (Layer 3) + HDBSCAN Clustering + Combine All
# 
# Enhanced with:
# - Original cross-user embedding similarity
# - **HDBSCAN clustering** on embeddings (users in same cluster as bots are more suspicious)
# - **Temporal pattern clustering** on hour distributions
# - Combine ALL feature layers into final feature matrices


# ============================================================
# Layer 3: Cross-user embedding similarity features
# ============================================================
def extract_cross_user_features(mean_embeddings):
    """Compute how similar each user's avg embedding is to all other users."""
    user_ids = list(mean_embeddings.keys())
    if len(user_ids) < 2:
        return pd.DataFrame(columns=['user_id', 'cross_avg_sim', 'cross_max_sim',
                                      'cross_min_sim', 'cross_std_sim'])

    emb_matrix = np.array([mean_embeddings[uid] for uid in user_ids])
    sim_matrix = cosine_similarity(emb_matrix)

    results = []
    for i, uid in enumerate(user_ids):
        sims = np.delete(sim_matrix[i], i)  # Exclude self-similarity
        results.append({
            'user_id': uid,
            'cross_avg_sim': float(np.mean(sims)),
            'cross_max_sim': float(np.max(sims)),
            'cross_min_sim': float(np.min(sims)),
            'cross_std_sim': float(np.std(sims)),
        })

    return pd.DataFrame(results)


# ============================================================
# Layer 3b: HDBSCAN Clustering Features (NEW)
# ============================================================
def extract_clustering_features(mean_embeddings, posts_df, users_df):
    """
    Cluster users using HDBSCAN on embeddings and temporal patterns.
    Returns per-user cluster membership and cluster-level statistics.
    """
    user_ids = list(mean_embeddings.keys())
    results = {uid: {} for uid in users_df['id'].tolist()}

    # --- Embedding-based HDBSCAN Clustering ---
    if len(user_ids) >= 10:
        emb_matrix = np.array([mean_embeddings[uid] for uid in user_ids])

        # Normalize for better clustering
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb_matrix)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean')
        cluster_labels = clusterer.fit_predict(emb_scaled)

        for i, uid in enumerate(user_ids):
            results[uid]['emb_cluster_id'] = int(cluster_labels[i])
            results[uid]['emb_cluster_prob'] = float(clusterer.probabilities_[i])
            # Cluster size (how many users in same cluster)
            if cluster_labels[i] >= 0:
                results[uid]['emb_cluster_size'] = int((cluster_labels == cluster_labels[i]).sum())
            else:
                results[uid]['emb_cluster_size'] = 0  # Noise point
            results[uid]['emb_is_noise'] = int(cluster_labels[i] == -1)
    else:
        for uid in user_ids:
            results[uid]['emb_cluster_id'] = -1
            results[uid]['emb_cluster_prob'] = 0.0
            results[uid]['emb_cluster_size'] = 0
            results[uid]['emb_is_noise'] = 1

    # --- Temporal Pattern Clustering ---
    # Build 24-dim hour distribution vector per user
    temporal_vectors = {}
    for _, user in users_df.iterrows():
        uid = user['id']
        user_posts_local = posts_df[posts_df['author_id'] == uid]
        if len(user_posts_local) >= 3:
            try:
                ts = pd.to_datetime(user_posts_local['created_at'])
                hour_dist = np.zeros(24)
                for h in ts.dt.hour:
                    hour_dist[h] += 1
                hour_dist /= max(hour_dist.sum(), 1)
                temporal_vectors[uid] = hour_dist
            except:
                temporal_vectors[uid] = np.zeros(24)
        else:
            temporal_vectors[uid] = np.zeros(24)

    temp_user_ids = list(temporal_vectors.keys())
    if len(temp_user_ids) >= 10:
        temp_matrix = np.array([temporal_vectors[uid] for uid in temp_user_ids])

        temp_clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean')
        temp_labels = temp_clusterer.fit_predict(temp_matrix)

        for i, uid in enumerate(temp_user_ids):
            results[uid]['temp_cluster_id'] = int(temp_labels[i])
            results[uid]['temp_cluster_prob'] = float(temp_clusterer.probabilities_[i])
            if temp_labels[i] >= 0:
                results[uid]['temp_cluster_size'] = int((temp_labels == temp_labels[i]).sum())
            else:
                results[uid]['temp_cluster_size'] = 0
            results[uid]['temp_is_noise'] = int(temp_labels[i] == -1)
    else:
        for uid in temp_user_ids:
            results[uid]['temp_cluster_id'] = -1
            results[uid]['temp_cluster_prob'] = 0.0
            results[uid]['temp_cluster_size'] = 0
            results[uid]['temp_is_noise'] = 1

    # Fill defaults for any users missing features
    default_cluster = {
        'emb_cluster_id': -1, 'emb_cluster_prob': 0.0, 'emb_cluster_size': 0, 'emb_is_noise': 1,
        'temp_cluster_id': -1, 'temp_cluster_prob': 0.0, 'temp_cluster_size': 0, 'temp_is_noise': 1,
    }
    rows = []
    for uid in users_df['id'].tolist():
        row = {'user_id': uid}
        for k, v in default_cluster.items():
            row[k] = results.get(uid, {}).get(k, v)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Combine ALL features into final feature matrices
# ============================================================
feature_dfs = {}
for ds_id in datasets.keys():
    print(f"\nCombining features for dataset {ds_id}...")

    # Start with basic features
    df = basic_features[ds_id].copy()

    # Merge advanced temporal features (NEW)
    if ds_id in advanced_temporal_features:
        df = df.merge(advanced_temporal_features[ds_id], on='user_id', how='left')

    # Merge stylometry features
    if ds_id in stylometry_features:
        df = df.merge(stylometry_features[ds_id], on='user_id', how='left')

    # Merge mention network features (NEW v3)
    if ds_id in mention_features:
        df = df.merge(mention_features[ds_id], on='user_id', how='left')

    # Merge NLP features
    df = df.merge(nlp_features[ds_id], on='user_id', how='left')

    # Merge topic coherence features (NEW v3)
    if ds_id in topic_coherence_features:
        df = df.merge(topic_coherence_features[ds_id], on='user_id', how='left')

    # Merge sentiment features (NEW v3)
    if ds_id in sentiment_features:
        df = df.merge(sentiment_features[ds_id], on='user_id', how='left')

    # Cross-user features
    if ds_id in all_mean_embeddings and all_mean_embeddings[ds_id]:
        cross_feats = extract_cross_user_features(all_mean_embeddings[ds_id])
        df = df.merge(cross_feats, on='user_id', how='left')

    # HDBSCAN clustering features
    if ds_id in all_mean_embeddings and all_mean_embeddings[ds_id]:
        cluster_feats = extract_clustering_features(
            all_mean_embeddings[ds_id], datasets[ds_id]['posts'], datasets[ds_id]['users'])
        df = df.merge(cluster_feats, on='user_id', how='left')

    # Fill NaN
    df = df.fillna(0)
    feature_dfs[ds_id] = df
    print(f"  Final feature shape: {df.shape}")

# List all feature columns (exclude metadata columns)
meta_cols = ['user_id', 'is_bot']
feature_cols = [c for c in feature_dfs[30].columns if c not in meta_cols]
print(f"\nTotal features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# ============================================================
# Free GPU memory: NLP models no longer needed for training
# They will be reloaded before detect_bots() if needed
# ============================================================
print("\n[GPU] Releasing NLP models from GPU to free VRAM for tree model training...")
import gc

# Delete perplexity models
for lang_code, ppl_info in ppl_models.items():
    ppl_info['model'].cpu()
    del ppl_info['model']
del ppl_models
print("  Perplexity models released.")

# Delete sentiment pipeline
del sentiment_pipe
print("  Sentiment pipeline released.")

# Delete sentence transformer
embed_model.cpu()
del embed_model
print("  Sentence Transformer released.")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    total_mem = torch.cuda.mem_get_info()[1] / 1024**3
    print(f"  GPU memory after cleanup: {free_mem:.1f} GB free / {total_mem:.1f} GB total")
print("[GPU] Cleanup complete. Ready for tree model training.\n")


# ## Section 5: Model Training & Stacking Ensemble (Enhanced)
# 
# Enhanced with:
# - **Tuned scale_pos_weight**: Adjusted based on competition scoring (+4 TP, -1 FN, -2 FP)
# - **Stacking Ensemble**: Layer 1 (XGBoost + LightGBM + CatBoost) -> Layer 2 (LogisticRegression)
# - **Fine-grained threshold search**: Two-pass coarse-to-fine search for optimal threshold


# ============================================================
# Competition Scoring Function
# ============================================================
def competition_score(y_true, y_pred):
    """Calculate the competition score: +4 TP, -1 FN, -2 FP."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    score = 4 * tp - 1 * fn - 2 * fp
    return score, {'tp': int(tp), 'fn': int(fn), 'fp': int(fp), 'tn': int(tn)}


def find_optimal_threshold(y_true, y_proba):
    """
    Two-pass coarse-to-fine threshold search for competition score maximization.
    Pass 1: coarse search (step=0.05) to find approximate region
    Pass 2: fine search (step=0.002) around the best region
    """
    best_score = -np.inf
    best_threshold = 0.5
    best_details = {}

    # Pass 1: Coarse search
    for threshold in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        score, details = competition_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_details = details

    # Pass 2: Fine search around the best threshold from pass 1
    fine_start = max(0.01, best_threshold - 0.08)
    fine_end = min(0.99, best_threshold + 0.08)
    for threshold in np.arange(fine_start, fine_end, 0.002):
        y_pred = (y_proba >= threshold).astype(int)
        score, details = competition_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_details = details

    return best_threshold, best_score, best_details


# ============================================================
# Custom Asymmetric Loss for Competition Scoring (CORE CHANGE)
# ============================================================
# Competition scoring: +4 TP, -1 FN, -2 FP, 0 TN
#
# Cost analysis (from the model's perspective):
#   - Missing a bot (FN): lose +4 TP reward AND get -1 FN penalty = 5 point swing
#   - False alarm  (FP): get -2 FP penalty                       = 2 point swing
#   - Ratio: 5:2 = 2.5x bias towards catching bots (recall)
#
# Instead of using scale_pos_weight (indirect), we directly encode
# these asymmetric costs into the gradient/hessian of the loss function.
# This makes the model learn the EXACT cost structure during training.

W_POS = 5.0  # Weight for bot class (positive): TP value (4) + FN cost (1)
W_NEG = 2.0  # Weight for human class (negative): FP cost (2)


def _safe_sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -500, 500)))


def competition_loss_xgb(y_true, y_pred):
    """
    Custom asymmetric weighted cross-entropy for XGBoost (sklearn API).

    Directly encodes the competition's asymmetric scoring into the loss:
    - Bot samples (y=1) get weight W_POS=5 (high cost of missing)
    - Human samples (y=0) get weight W_NEG=2 (cost of false alarm)

    Args:
        y_true: true labels (0 or 1)
        y_pred: raw predictions (logits, before sigmoid)
    Returns:
        (gradient, hessian) tuple
    """
    p = _safe_sigmoid(y_pred)
    w = np.where(y_true == 1, W_POS, W_NEG)
    # Gradient of weighted binary cross-entropy w.r.t. logit
    grad = w * (p - y_true)
    # Hessian of weighted binary cross-entropy w.r.t. logit
    hess = w * np.maximum(p * (1.0 - p), 1e-7)
    return grad, hess


def competition_loss_lgb(y_true, y_pred):
    """
    Custom asymmetric weighted cross-entropy for LightGBM (sklearn API).
    Same math as XGBoost version, separate function for clarity.
    """
    p = _safe_sigmoid(y_pred)
    w = np.where(y_true == 1, W_POS, W_NEG)
    grad = w * (p - y_true)
    hess = w * np.maximum(p * (1.0 - p), 1e-7)
    return grad, hess


def train_models(X_train, y_train, params=None):
    """
    Train XGBoost, LightGBM, and CatBoost with CUSTOM ASYMMETRIC LOSS.
    Layer 1 of the stacking ensemble.

    Key change: replaced scale_pos_weight with custom objective functions
    that directly encode the competition's asymmetric scoring (4*TP - FN - 2*FP).

    Args:
        X_train: training features
        y_train: training labels
        params: optional dict of hyperparameters from Optuna optimization.
                If None, uses default hyperparameters.
    """
    # Default hyperparameters (used when params is None)
    if params is None:
        params = {
            'n_estimators': 600,
            'max_depth': 5,
            'learning_rate': 0.025,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.1,
        }

    # XGBoost with custom asymmetric loss
    # NOTE: removed scale_pos_weight and eval_metric='logloss'
    #       because the custom objective handles asymmetric costs directly
    xgb_model = xgb.XGBClassifier(
        objective=competition_loss_xgb,
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        gamma=params['gamma'],
        device=XGB_DEVICE,  # GPU acceleration when available
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)

    # LightGBM with custom asymmetric loss
    lgb_model = lgb.LGBMClassifier(
        objective=competition_loss_lgb,
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    # CatBoost with equivalent asymmetric class weights
    # CatBoost doesn't support custom obj as easily via sklearn API,
    # so we use class_weights with the same 5:2 ratio
    cb_model = CatBoostClassifier(
        iterations=params['n_estimators'],
        depth=min(params['max_depth'], 10),  # CatBoost max depth capped at 10
        learning_rate=params['learning_rate'],
        class_weights={0: 1.0, 1: W_POS / W_NEG},  # {0: 1.0, 1: 2.5}
        task_type=CB_TASK_TYPE,  # GPU acceleration when available
        random_seed=42, verbose=0
    )
    cb_model.fit(X_train, y_train)

    return {'xgb': xgb_model, 'lgb': lgb_model, 'cb': cb_model}


def get_base_model_probas(models, X):
    """
    Get probability predictions from each base model (for stacking).

    IMPORTANT: With custom objectives, XGBoost/LightGBM's predict_proba()
    returns RAW LOGITS instead of probabilities (they don't know the link
    function for custom objectives). We extract raw predictions and apply
    sigmoid manually.
    CatBoost with class_weights uses standard objective, so predict_proba works.
    """
    probas = {}
    for name, model in models.items():
        if name == 'xgb':
            # XGBoost: get raw margin via booster and apply sigmoid
            dmat = xgb.DMatrix(X)
            raw = model.get_booster().predict(dmat, output_margin=True)
            probas[name] = _safe_sigmoid(raw)
        elif name == 'lgb':
            # LightGBM: get raw score and apply sigmoid
            raw = model.predict(X, raw_score=True)
            probas[name] = _safe_sigmoid(raw)
        else:
            # CatBoost: predict_proba works correctly with class_weights
            probas[name] = model.predict_proba(X)[:, 1]
    return probas


def train_stacking_meta(models, X_train, y_train):
    """
    Train Layer 2 meta-learner using cross-validated predictions from Layer 1.
    Uses LogisticRegression as the meta-learner to learn optimal model weighting.
    """
    from sklearn.model_selection import StratifiedKFold

    n_models = len(models)
    meta_features = np.zeros((len(X_train), n_models))

    # Generate out-of-fold predictions for meta-training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr = y_train[tr_idx]

        # Train fold models
        fold_models = train_models(X_tr, y_tr)

        # Get OOF predictions (using get_base_model_probas for proper sigmoid handling)
        fold_probas = get_base_model_probas(fold_models, X_val)
        for i, name in enumerate(fold_models.keys()):
            meta_features[val_idx, i] = fold_probas[name]

    # Train meta-learner on OOF predictions
    meta_model = LogisticRegressionCV(
        Cs=10, cv=3, scoring='roc_auc',
        class_weight='balanced', random_state=42, max_iter=1000
    )
    meta_model.fit(meta_features, y_train)

    print(f"  Stacking meta-learner trained. Coefficients: {dict(zip(models.keys(), meta_model.coef_[0]))}")
    return meta_model


def ensemble_predict_proba(models, X, meta_model=None):
    """
    Get ensemble probability.
    If meta_model is provided: use stacking (Layer 2 meta-learner).
    Otherwise: fall back to simple average.
    """
    base_probas = get_base_model_probas(models, X)
    probas_array = np.column_stack(list(base_probas.values()))

    if meta_model is not None:
        return meta_model.predict_proba(probas_array)[:, 1]
    else:
        return np.mean(probas_array, axis=1)


def evaluate_on_test(models, X_test, y_test, dataset_name="", meta_model=None, user_ids=None):
    """Evaluate ensemble on test set with optimal threshold.
    If user_ids is provided, also prints which users were misclassified (FP/FN).
    """
    proba = ensemble_predict_proba(models, X_test, meta_model=meta_model)
    threshold, score, details = find_optimal_threshold(y_test, proba)

    y_pred = (proba >= threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"Evaluation: {dataset_name}")
    print(f"{'='*50}")
    print(f"Optimal threshold: {threshold:.2f}")
    print(f"Competition score: {score}")
    print(f"Details: TP={details['tp']}, FN={details['fn']}, FP={details['fp']}, TN={details['tn']}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Bot']))

    # Print misclassified users for error analysis
    if user_ids is not None:
        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred)
        uids = np.array(user_ids)

        # False Negatives: real bots missed (predicted human)
        fn_mask = (y_test_arr == 1) & (y_pred_arr == 0)
        if fn_mask.any():
            print(f"  [FN] Missed bots (real bot, predicted human) — cost: -1 each:")
            for uid, p in zip(uids[fn_mask], proba[fn_mask]):
                print(f"    user_id={uid}  proba={p:.4f}")

        # False Positives: humans misclassified as bots
        fp_mask = (y_test_arr == 0) & (y_pred_arr == 1)
        if fp_mask.any():
            print(f"  [FP] False alarms (real human, predicted bot) — cost: -2 each:")
            for uid, p in zip(uids[fp_mask], proba[fp_mask]):
                print(f"    user_id={uid}  proba={p:.4f}")

        if not fn_mask.any() and not fp_mask.any():
            print("  [PERFECT] No misclassifications!")

    return threshold, score, proba

print("Training functions defined.")


# ## Section 6: Cross-Validation Between Datasets
# 
# - English: Train on 30 -> Test on 32, then Train on 32 -> Test on 30
# - French: Train on 31 -> Test on 33, then Train on 33 -> Test on 31


# ============================================================
# Optuna Bayesian Hyperparameter Optimization
# ============================================================
# Replaces manual grid/random search with efficient TPE (Tree-structured
# Parzen Estimator) sampler. The competition score (+4 TP, -1 FN, -2 FP)
# is used directly as the optimization objective.

def optuna_optimize_hyperparams(X_train, y_train, n_trials=200, n_inner_folds=3):
    """
    Use Optuna Bayesian optimization (TPE) to find the best hyperparameters
    for the XGBoost + LightGBM + CatBoost stacking ensemble.

    The objective maximizes the average competition score across inner CV folds.

    Args:
        X_train: training feature matrix (numpy array)
        y_train: training labels (numpy array)
        n_trials: number of Optuna trials (default 200, increase for better results)
        n_inner_folds: number of inner CV folds for evaluation (default 3)

    Returns:
        optuna.Study object (access best_params, best_value, etc.)
    """

    def objective(trial):
        # Search space aligned with bot_detector.py (proven ranges for ~250 user datasets)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        }

        # Inner cross-validation to evaluate this set of hyperparameters
        skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
        fold_scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            # Train all base models with the suggested hyperparameters
            fold_models = train_models(X_tr, y_tr, params=params)

            # Get ensemble probability (simple average, no meta-learner in inner loop
            # for speed -- the meta-learner will be trained after optimization)
            probas = get_base_model_probas(fold_models, X_val)
            avg_proba = np.mean(np.column_stack(list(probas.values())), axis=1)

            # Evaluate with competition score using optimal threshold
            _, score, _ = find_optimal_threshold(y_val, avg_proba)
            fold_scores.append(score)

        return np.mean(fold_scores)

    # Create Optuna study with TPE sampler (Bayesian optimization)
    # n_startup_trials=20: let TPE explore more before pruning (better for noisy objectives)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )

    # Print optimization results
    print(f"\n  Optuna optimization complete ({n_trials} trials)")
    print(f"  Best competition score (inner CV avg): {study.best_value:.2f}")
    print(f"  Best hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

    return study


# ============================================================
# Save / Load Hyperparameters (JSON)
# ============================================================
# On Google Colab, saves to Google Drive so hyperparameters persist
# across sessions. Locally, saves to saved_hyperparams/ in the
# current working directory.
# ============================================================
import os

# Hyperparameters saved to per-run directory (with timestamp) AND shared directory
# Per-run: runs/<RUN_NAME>/hyperparams/       (immutable snapshot of this run)
# Shared:  saved_hyperparams_max/             (latest version, for reuse by future runs)
# NOTE: bot_detector.py uses saved_hyperparams/ — kept separate to avoid conflicts

IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
    HYPERPARAMS_DIR = "/content/drive/MyDrive/bot_or_not/saved_hyperparams_max"
    print(f"[Colab] Hyperparameters will be saved to Google Drive: {HYPERPARAMS_DIR}")
else:
    HYPERPARAMS_DIR = os.path.join(DATA_DIR, "saved_hyperparams_max")
    print(f"[Local] Hyperparameters will be saved to: {HYPERPARAMS_DIR}")

# Per-run hyperparams directory (immutable snapshot with timestamp)
RUN_HYPERPARAMS_DIR = os.path.join(LOG_DIR, "hyperparams")
os.makedirs(RUN_HYPERPARAMS_DIR, exist_ok=True)
print(f"[Run]   Hyperparameters snapshot: {RUN_HYPERPARAMS_DIR}")

def save_hyperparams(params, name, score=None, directory=HYPERPARAMS_DIR):
    """
    Save hyperparameters to BOTH shared and per-run directories.

    Shared dir (saved_hyperparams/):  latest version, for loading in future runs.
    Run dir (runs/<timestamp>/hyperparams/): immutable snapshot with run timestamp.

    Args:
        params: dict of hyperparameters
        name: identifier string, e.g. 'cv_30_32' or 'best_english'
        score: optional competition score to save alongside
        directory: shared folder (default: saved_hyperparams/)
    """
    payload = {
        "hyperparameters": params,
        "score": score,
        "run_name": RUN_NAME,
        "saved_at": pd.Timestamp.now().isoformat(),
    }

    # 1. Save to shared directory (latest, overwritten each run)
    os.makedirs(directory, exist_ok=True)
    shared_path = os.path.join(directory, f"{name}.json")
    with open(shared_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  [SAVED] Hyperparams (shared) -> {shared_path}")

    # 2. Save to per-run directory (immutable, with run timestamp in folder name)
    run_path = os.path.join(RUN_HYPERPARAMS_DIR, f"{name}.json")
    with open(run_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  [SAVED] Hyperparams (run)    -> {run_path}")

    return shared_path


def load_hyperparams(name, directory=HYPERPARAMS_DIR):
    """
    Load previously saved hyperparameters from a JSON file.

    Args:
        name: identifier string used when saving
        directory: folder to look in (default: saved_hyperparams/)

    Returns:
        dict of hyperparameters, or None if file not found
    """
    filepath = os.path.join(directory, f"{name}.json")
    if not os.path.exists(filepath):
        print(f"  [WARNING] No saved hyperparameters found at {filepath}")
        return None
    with open(filepath, 'r') as f:
        payload = json.load(f)
    params = payload["hyperparameters"]
    score = payload.get("score")
    saved_at = payload.get("saved_at", "unknown")
    print(f"  [LOADED] Hyperparameters from {filepath} (score={score}, saved_at={saved_at})")
    # Restore int types for integer hyperparameters
    int_keys = ['n_estimators', 'max_depth', 'min_child_weight']
    for k in int_keys:
        if k in params:
            params[k] = int(params[k])
    return params


def list_saved_hyperparams(directory=HYPERPARAMS_DIR):
    """List all saved hyperparameter files."""
    if not os.path.exists(directory):
        print("No saved hyperparameters directory found.")
        return []
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        print("No saved hyperparameters found.")
        return []
    print(f"Saved hyperparameters in '{directory}/':")
    for f in sorted(files):
        filepath = os.path.join(directory, f)
        with open(filepath, 'r') as fh:
            payload = json.load(fh)
        score = payload.get("score", "N/A")
        saved_at = payload.get("saved_at", "unknown")
        print(f"  - {f}: score={score}, saved_at={saved_at}")
    return files


print("Optuna hyperparameter optimization function defined.")
print("Hyperparameter save/load utilities defined.")



# Cross-validation pairs: (train_id, test_id, language)
cv_pairs = [
    (30, 32, 'English'),
    (32, 30, 'English'),
    (31, 33, 'French'),
    (33, 31, 'French'),
]

# ============================================================
# Optuna Configuration — BEST RESULT MODE
# ============================================================
USE_OPTUNA = False         # Skip Optuna, use saved hyperparams
OPTUNA_N_TRIALS = 60       # 60 trials: good balance (best usually found <20 trials)
OPTUNA_INNER_FOLDS = 3     # 3-fold: more stable evaluation on ~250 users

# Load saved hyperparams from saved_hyperparams_max/ (skip Optuna)
LOAD_SAVED_HYPERPARAMS = True

cv_results = {}
trained_models = {}
trained_meta_models = {}
optuna_studies = {}  # Store Optuna studies for analysis
cv_pairs_lang = {(tr, te): lang for tr, te, lang in cv_pairs}  # Quick lang lookup

for train_id, test_id, lang in cv_pairs:
    print(f"\n{'#'*60}")
    print(f"Training on Dataset {train_id} -> Testing on Dataset {test_id} ({lang})")
    print(f"{'#'*60}")

    train_df = feature_dfs[train_id]
    test_df = feature_dfs[test_id]

    # Ensure consistent feature columns
    common_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]

    X_train = train_df[common_cols].values
    y_train = train_df['is_bot'].values
    X_test = test_df[common_cols].values
    y_test = test_df['is_bot'].values

    # ---- Hyperparameter Selection ----
    best_params = None
    param_name = f"cv_{train_id}_{test_id}"

    # Step 1: Try loading saved hyperparameters
    if LOAD_SAVED_HYPERPARAMS:
        best_params = load_hyperparams(param_name)
        if best_params is not None:
            print(f"  Using saved hyperparameters for {param_name}")

    # Step 1b: Reuse Optuna params from same-language pair (avoid redundant optimization)
    if best_params is None:
        for (prev_tr, prev_te), prev_study in optuna_studies.items():
            prev_lang = cv_pairs_lang.get((prev_tr, prev_te))
            if prev_lang == lang:
                best_params = prev_study.best_params
                print(f"  Reusing Optuna params from ({prev_tr}->{prev_te}) for same language ({lang})")
                save_hyperparams(best_params, param_name, score=prev_study.best_value)
                break

    # Step 2: If no saved/reused params, try Optuna (only if enabled)
    if best_params is None and USE_OPTUNA:
        print(f"  Running Optuna hyperparameter optimization ({OPTUNA_N_TRIALS} trials)...")
        study = optuna_optimize_hyperparams(
            X_train, y_train,
            n_trials=OPTUNA_N_TRIALS,
            n_inner_folds=OPTUNA_INNER_FOLDS
        )
        best_params = study.best_params
        optuna_studies[(train_id, test_id)] = study
        save_hyperparams(best_params, param_name, score=study.best_value)

    # Step 3: If still no params, use defaults (best_params=None -> train_models uses defaults)
    if best_params is None:
        print(f"  Using default hyperparameters (no saved params, Optuna disabled)")

    # Train Layer 1 base models (with Optuna-optimized or default hyperparameters)
    print("  Training Layer 1 base models...")
    models = train_models(X_train, y_train, params=best_params)
    trained_models[(train_id, test_id)] = models

    # Train Layer 2 meta-learner (stacking)
    print("  Training Layer 2 meta-learner (stacking)...")
    meta_model = train_stacking_meta(models, X_train, y_train)
    trained_meta_models[(train_id, test_id)] = meta_model

    # Evaluate with stacking (pass user_ids for error analysis)
    test_user_ids = test_df['user_id'].values
    threshold, score, proba = evaluate_on_test(
        models, X_test, y_test, f"Train {train_id} -> Test {test_id} ({lang})",
        meta_model=meta_model, user_ids=test_user_ids)

    cv_results[(train_id, test_id)] = {
        'threshold': threshold, 'score': score,
        'lang': lang, 'proba': proba
    }

# Summary
print(f"\n{'='*60}")
print("CROSS-VALIDATION SUMMARY (with Stacking Ensemble)")
print(f"{'='*60}")
for (tr, te), res in cv_results.items():
    print(f"  Train {tr} -> Test {te} ({res['lang']}): "
          f"Score = {res['score']}, Threshold = {res['threshold']:.2f}")



# Feature importance analysis (text only, skip plot to save time)
first_key = list(trained_models.keys())[0]
xgb_model = trained_models[first_key]['xgb']
common_cols = [c for c in feature_cols if c in feature_dfs[first_key[0]].columns]

importance = pd.DataFrame({
    'feature': common_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 features:")
print(importance.head(15).to_string(index=False))


# ## Section 7: Train Final Models & Build Submission Pipeline
# 
# Train on ALL available practice data per language, then use for the final evaluation.


# ============================================================
# Train final models on ALL practice data per language (with Stacking)
# ============================================================
# Compute average optimal threshold from cross-validation
en_thresholds = [v['threshold'] for (tr, te), v in cv_results.items() if v['lang'] == 'English']
fr_thresholds = [v['threshold'] for (tr, te), v in cv_results.items() if v['lang'] == 'French']
final_en_threshold = np.mean(en_thresholds)
final_fr_threshold = np.mean(fr_thresholds)

print(f"Final English threshold: {final_en_threshold:.2f}")
print(f"Final French threshold: {final_fr_threshold:.2f}")

# Combine English datasets (30 + 32)
en_train = pd.concat([feature_dfs[30], feature_dfs[32]], ignore_index=True)
common_cols_en = [c for c in feature_cols if c in en_train.columns]
X_en = en_train[common_cols_en].values
y_en = en_train['is_bot'].values

# Select best Optuna params for each language
# Priority: 1) Load from saved files  2) From Optuna studies in memory  3) None (use defaults)
best_en_params = None
best_fr_params = None

if LOAD_SAVED_HYPERPARAMS:
    # Try to load saved "best" params for final models
    best_en_params = load_hyperparams("best_english")
    best_fr_params = load_hyperparams("best_french")

if best_en_params is None and USE_OPTUNA and optuna_studies:
    # English: pick params from the CV pair with the best study score
    en_studies = {k: v for k, v in optuna_studies.items() if cv_results[k]['lang'] == 'English'}
    if en_studies:
        best_en_key = max(en_studies, key=lambda k: en_studies[k].best_value)
        best_en_params = en_studies[best_en_key].best_params
        print(f"  Using Optuna params from study ({best_en_key}) for final English model")
        # Save the best English params for future reuse
        save_hyperparams(best_en_params, "best_english", score=en_studies[best_en_key].best_value)

if best_fr_params is None and USE_OPTUNA and optuna_studies:
    # French: pick params from the CV pair with the best study score
    fr_studies = {k: v for k, v in optuna_studies.items() if cv_results[k]['lang'] == 'French'}
    if fr_studies:
        best_fr_key = max(fr_studies, key=lambda k: fr_studies[k].best_value)
        best_fr_params = fr_studies[best_fr_key].best_params
        print(f"  Using Optuna params from study ({best_fr_key}) for final French model")
        # Save the best French params for future reuse
        save_hyperparams(best_fr_params, "best_french", score=fr_studies[best_fr_key].best_value)

print(f"\nTraining final English model on {len(X_en)} users ({y_en.sum()} bots)...")
final_en_models = train_models(X_en, y_en, params=best_en_params)
print("  Training English stacking meta-learner...")
final_en_meta = train_stacking_meta(final_en_models, X_en, y_en)

# Combine French datasets (31 + 33)
fr_train = pd.concat([feature_dfs[31], feature_dfs[33]], ignore_index=True)
common_cols_fr = [c for c in feature_cols if c in fr_train.columns]
X_fr = fr_train[common_cols_fr].values
y_fr = fr_train['is_bot'].values

print(f"Training final French model on {len(X_fr)} users ({y_fr.sum()} bots)...")
final_fr_models = train_models(X_fr, y_fr, params=best_fr_params)
print("  Training French stacking meta-learner...")
final_fr_meta = train_stacking_meta(final_fr_models, X_fr, y_fr)

print("\nFinal models (with stacking) trained!")



# ============================================================
# POST-PROCESSING: Rule-based rescue for missed bots
# ============================================================
def post_process_rescue(posts_df, users_df, proba, threshold, ds_lang):
    """
    Post-processing step to rescue bots that the main model missed.
    Only examines users the main model scored BELOW threshold (predicted human).
    Returns list of indices to flip from human -> bot.

    Rules (ultra-conservative — only fire on near-certain bot patterns):
      Rule 1: High question ratio (>=0.8) + high consecutive streak (>=0.6)
              Catches "confused persona" bots (e.g. SirReginald73, grandpapa_uk)
              CV-validated: 2 rescued FN, 0 new FP, +10 score improvement
    """
    rescued_indices = []
    rescued_details = []

    for idx in range(len(proba)):
        if proba[idx] >= threshold:
            continue  # Already predicted as bot, skip

        uid = users_df.iloc[idx]['id'] if idx < len(users_df) else None
        user_posts = posts_df[posts_df['author_id'] == uid] if uid else pd.DataFrame()
        texts = user_posts['text'].tolist() if len(user_posts) > 0 else []
        n_posts = len(texts)

        if n_posts < 3:
            continue

        # --- Compute lightweight features ---
        question_flags = [1 if ('?' in t or '？' in t) else 0 for t in texts]
        question_ratio = sum(question_flags) / n_posts

        # Longest consecutive question streak
        max_streak, cur_streak = 0, 0
        for qf in question_flags:
            if qf:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 0
        interrogative_monotony = max_streak / n_posts

        # Same-second burst: count max posts at identical timestamp
        timestamps = user_posts['created_at'].tolist() if len(user_posts) > 0 else []
        from collections import Counter as _Counter
        ts_counts = _Counter(timestamps)
        max_same_ts = max(ts_counts.values()) if ts_counts else 0

        # --- Apply rules ---
        rule_triggered = None

        # Rule 1: Almost all posts are questions (catches SirReginald73, grandpapa_uk)
        # CV-validated: 0 false positives across all 4 datasets
        if question_ratio >= 0.8 and interrogative_monotony >= 0.6:
            rule_triggered = f"R1:question_ratio={question_ratio:.2f},monotony={interrogative_monotony:.2f}"

        # Rule 4: Same-second batch posting (catches lilie_paris, papycharles)
        # No human can post 4+ tweets at the exact same second
        elif max_same_ts >= 4:
            rule_triggered = f"R4:max_same_timestamp={max_same_ts}"

        if rule_triggered:
            rescued_indices.append(idx)
            rescued_details.append((uid, proba[idx], rule_triggered))

    if rescued_details:
        print(f"\n  [POST-PROCESS] Rescued {len(rescued_details)} additional bots:")
        for uid, p, rule in rescued_details:
            print(f"    {uid}  max_proba={p:.4f}  rule={rule}")
    else:
        print(f"\n  [POST-PROCESS] No additional bots rescued.")

    return rescued_indices


# ============================================================
# FINAL PIPELINE: Process a new dataset and output bot IDs (Enhanced v3)
# ============================================================
def detect_bots(json_path, models, threshold, feature_columns,
                embed_model, ppl_models_dict, sentiment_pipe,
                meta_model=None,
                team_name="myteam", lang="en",
                output_dir=LOG_DIR):
    """
    Complete bot detection pipeline for a new dataset (Enhanced v3).
    Includes: basic + advanced temporal + stylometry + mention network +
              NLP (multilingual perplexity) + topic coherence + sentiment +
              cross-user + clustering features.
    Uses stacking ensemble if meta_model is provided.
    Input: path to dataset JSON
    Output: saves bot IDs to a text file and returns the list
    """
    print(f"{'='*60}")
    print(f"PROCESSING: {json_path}")
    print(f"{'='*60}")

    # Step 1: Load data
    print("[1/10] Loading data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    posts_df = pd.DataFrame(data['posts'])
    users_df = pd.DataFrame(data['users'])
    ds_lang = data.get('lang', lang)
    print(f"  {len(users_df)} users, {len(posts_df)} posts, language: {ds_lang}")

    # Step 2: Basic features
    print("[2/10] Extracting basic features...")
    basic = extract_all_basic_features(posts_df, users_df)

    # Step 3: Advanced temporal features (with second-level periodicity)
    print("[3/10] Extracting advanced temporal features...")
    temporal_results = []
    for _, user in users_df.iterrows():
        uid = user['id']
        user_posts = posts_df[posts_df['author_id'] == uid]
        f = {'user_id': uid}
        f.update(extract_advanced_temporal_features(user_posts))
        temporal_results.append(f)
    temporal_feats = pd.DataFrame(temporal_results)

    # Step 4: Stylometry features
    print("[4/10] Extracting stylometry features...")
    stylo_results = []
    for _, user in users_df.iterrows():
        uid = user['id']
        user_posts = posts_df[posts_df['author_id'] == uid]
        f = {'user_id': uid}
        f.update(extract_stylometry_features(user_posts))
        stylo_results.append(f)
    stylo_feats = pd.DataFrame(stylo_results)

    # Step 5: Mention network features (NEW v3)
    print("[5/10] Extracting mention network features...")
    mention_by_user, username_to_id, mentioned_by = build_mention_graph(posts_df, users_df)
    all_user_ids_set = set(users_df['id'])
    mention_results = []
    for _, user in users_df.iterrows():
        uid = user['id']
        f = {'user_id': uid}
        f.update(extract_mention_features(
            uid, mention_by_user, username_to_id, mentioned_by, all_user_ids_set))
        mention_results.append(f)
    mention_feats = pd.DataFrame(mention_results)

    # Step 6: NLP features (multilingual perplexity)
    print(f"[6/10] Computing NLP features (perplexity model: {PPL_MODELS.get(ds_lang, 'gpt2')})...")
    emb_feats, mean_embs = extract_embedding_features(posts_df, users_df, model=embed_model)
    ppl_info = ppl_models_dict.get(ds_lang, ppl_models_dict.get('en'))
    ppl_feats = extract_perplexity_features(
        posts_df, users_df,
        ppl_info['model'], ppl_info['tokenizer'], ppl_info['device'])
    nlp_feat = emb_feats.merge(ppl_feats, on='user_id', how='outer')

    # Step 7: Topic coherence features (NEW v3)
    print("[7/10] Computing topic coherence features...")
    topic_feats = extract_topic_coherence_features(posts_df, users_df, embed_model)

    # Step 8: Sentiment features (NEW v3)
    print("[8/10] Computing sentiment features...")
    sent_feats = extract_sentiment_features(posts_df, users_df, sentiment_pipe)

    # Step 9: Cross-user + clustering features
    print("[9/10] Computing cross-user + clustering features...")
    cross_feats = extract_cross_user_features(mean_embs) if mean_embs else pd.DataFrame()
    cluster_feats = extract_clustering_features(mean_embs, posts_df, users_df) if mean_embs else pd.DataFrame()

    # Combine all features
    df = basic.copy()
    df = df.merge(temporal_feats, on='user_id', how='left')
    df = df.merge(stylo_feats, on='user_id', how='left')
    df = df.merge(mention_feats, on='user_id', how='left')
    df = df.merge(nlp_feat, on='user_id', how='left')
    df = df.merge(topic_feats, on='user_id', how='left')
    df = df.merge(sent_feats, on='user_id', how='left')
    if not cross_feats.empty:
        df = df.merge(cross_feats, on='user_id', how='left')
    if not cluster_feats.empty:
        df = df.merge(cluster_feats, on='user_id', how='left')
    df = df.fillna(0)

    # Ensure feature columns match training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_columns].values

    # Step 10: Predict (with stacking if available)
    print("[10/11] Predicting (stacking ensemble)...")
    proba = ensemble_predict_proba(models, X, meta_model=meta_model)
    predictions = (proba >= threshold).astype(int)

    # Step 11: Post-processing rescue
    print("[11/11] Post-processing rescue (rule-based)...")
    rescued_indices = post_process_rescue(posts_df, users_df, proba, threshold, ds_lang)
    for idx in rescued_indices:
        predictions[idx] = 1

    # Get bot user IDs
    bot_user_ids = df.loc[predictions == 1, 'user_id'].tolist()

    # Save to file (in output_dir for organized output)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{team_name}.detections.{ds_lang}.txt"
    output_filepath = os.path.join(output_dir, output_filename)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for uid in bot_user_ids:
            f.write(f"{uid}\n")

    print(f"\n  Detected {len(bot_user_ids)} bots out of {len(users_df)} users")
    print(f"  Results saved to: {output_filepath}")

    # Also print probability distribution for manual inspection
    print(f"\n  Probability distribution:")
    print(f"    Mean: {np.mean(proba):.4f}")
    print(f"    > 0.3: {(proba > 0.3).sum()} users")
    print(f"    > 0.5: {(proba > 0.5).sum()} users")
    print(f"    > 0.7: {(proba > 0.7).sum()} users")
    print(f"    > 0.9: {(proba > 0.9).sum()} users")

    return bot_user_ids, proba, df

print("Final pipeline (v3) defined.")


# ## Section 8: Validate Pipeline on Practice Data
# 
# Run the full pipeline on practice datasets to verify it works correctly and check the competition score.


# ============================================================
# Reload NLP models for detect_bots() pipeline
# (They were freed earlier to make room for tree model training)
# ============================================================
print("\n[GPU] Reloading NLP models for detect_bots pipeline...")

# Free tree model GPU memory first
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Reload sentence transformer
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("  Sentence Transformer reloaded.")

# Reload perplexity models
ppl_models = {}
for lang_code in ['en', 'fr']:
    ppl_m, ppl_t, ppl_d = load_perplexity_model(lang=lang_code)
    ppl_models[lang_code] = {'model': ppl_m, 'tokenizer': ppl_t, 'device': ppl_d}
print("  Perplexity models reloaded.")

# Reload sentiment pipeline
sentiment_pipe = load_sentiment_model()
print("  Sentiment pipeline reloaded.")

if torch.cuda.is_available():
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    total_mem = torch.cuda.mem_get_info()[1] / 1024**3
    print(f"  GPU memory: {free_mem:.1f} GB free / {total_mem:.1f} GB total")
print("[GPU] NLP models ready.\n")

# [SKIPPED] Section 8 validation — not needed for competition day, saves ~20 min
print("[SKIPPED] Section 8 validation (not needed for competition, saving time)")


# ## Section 9: Final Evaluation (Run on Competition Day)
# 
# On **Feb 14, 12:00 PM EST**, upload the new evaluation dataset JSON and run this cell.
# Change `TEAM_NAME` to your team name before running!


# ============================================================
# FINAL EVALUATION - CHANGE THESE TWO LINES BEFORE RUNNING
# ============================================================
TEAM_NAME = "Bot_killer_RX"  # <-- CHANGE THIS to your team name

# ============================================================
# Process BOTH evaluation datasets (34=English, 35=French)
# ============================================================
EVAL_FILES = [
    os.path.join(DATA_DIR, 'dataset.posts&users.34.json'),  # English
    os.path.join(DATA_DIR, 'dataset.posts&users.35.json'),  # French
]

all_eval_bots = []
for EVAL_JSON in EVAL_FILES:
    if not os.path.exists(EVAL_JSON):
        print(f"\n[SKIP] Not found: {EVAL_JSON}")
        continue

    with open(EVAL_JSON, 'r', encoding='utf-8') as f_eval:
        eval_meta = json.load(f_eval)
    eval_lang = eval_meta.get('lang', 'en')
    print(f"\nEvaluation dataset: {os.path.basename(EVAL_JSON)}, language: {eval_lang}")

    if eval_lang == 'fr':
        eval_models = final_fr_models
        eval_threshold = final_fr_threshold
        eval_meta_model = final_fr_meta
        eval_feature_cols = common_cols_fr
    else:
        eval_models = final_en_models
        eval_threshold = final_en_threshold
        eval_meta_model = final_en_meta
        eval_feature_cols = common_cols_en

    print(f"Using {'French' if eval_lang == 'fr' else 'English'} model, threshold={eval_threshold:.4f}")

    eval_bots, eval_proba, eval_df = detect_bots(
        EVAL_JSON,
        models=eval_models,
        threshold=eval_threshold,
        feature_columns=eval_feature_cols,
        embed_model=embed_model,
        ppl_models_dict=ppl_models,
        sentiment_pipe=sentiment_pipe,
        meta_model=eval_meta_model,
        team_name=TEAM_NAME, lang=eval_lang
    )
    all_eval_bots.extend(eval_bots)

    print(f"\n  DONE: {os.path.basename(EVAL_JSON)} -> {len(eval_bots)} bots detected")

# Save combined detection file
combined_path = os.path.join(LOG_DIR, f"{TEAM_NAME}.detections.all.txt")
with open(combined_path, 'w', encoding='utf-8') as f:
    for uid in all_eval_bots:
        f.write(f"{uid}\n")
print(f"\n{'='*60}")
print(f"ALL DETECTIONS COMBINED: {len(all_eval_bots)} bots -> {combined_path}")
print(f"Individual files also saved per language in {LOG_DIR}")
print(f"\nSubmit to: bot.or.not.competition.adm@gmail.com")
print(f"Deadline: Feb 14, 2026, 1:00 PM EST")
print(f"{'='*60}")



# Running locally - files are saved in the run output directory
for fname in [f"{TEAM_NAME}.detections.en.txt", f"{TEAM_NAME}.detections.fr.txt"]:
    fpath = os.path.join(LOG_DIR, fname)
    if os.path.exists(fpath):
        print(f"File saved: {fpath}")


# ============================================================
# Save run summary to JSON (machine-readable results)
# ============================================================
run_summary = {
    "run_name": RUN_NAME,
    "timestamp": RUN_TIMESTAMP,
    "total_features": len(feature_cols) if 'feature_cols' in dir() else 0,
    "optuna_enabled": USE_OPTUNA,
    "optuna_n_trials": OPTUNA_N_TRIALS if USE_OPTUNA else 0,
    "optuna_inner_folds": OPTUNA_INNER_FOLDS if USE_OPTUNA else 0,
    "cross_validation": {},
    "final_thresholds": {},
}

# Save CV results
if 'cv_results' in dir():
    for (tr, te), res in cv_results.items():
        key = f"train_{tr}_test_{te}"
        run_summary["cross_validation"][key] = {
            "lang": res['lang'],
            "score": int(res['score']),
            "threshold": round(float(res['threshold']), 4),
        }

# Save final thresholds
if 'final_en_threshold' in dir():
    run_summary["final_thresholds"]["en"] = round(float(final_en_threshold), 4)
if 'final_fr_threshold' in dir():
    run_summary["final_thresholds"]["fr"] = round(float(final_fr_threshold), 4)

# Save Optuna best scores
if 'optuna_studies' in dir() and optuna_studies:
    run_summary["optuna_best_scores"] = {}
    for (tr, te), study in optuna_studies.items():
        run_summary["optuna_best_scores"][f"train_{tr}_test_{te}"] = {
            "best_value": round(float(study.best_value), 2),
            "n_trials": len(study.trials),
        }

summary_path = os.path.join(LOG_DIR, f"{RUN_NAME}_summary.json")
with open(summary_path, 'w') as f:
    json.dump(run_summary, f, indent=2, default=str)
print(f"\n[SAVED] Run summary -> {summary_path}")

# Detection files are already saved directly to LOG_DIR by detect_bots()
for fname in [f"{TEAM_NAME}.detections.en.txt", f"{TEAM_NAME}.detections.fr.txt"]:
    fpath = os.path.join(LOG_DIR, fname)
    if os.path.exists(fpath):
        print(f"[SAVED] Detection file -> {fpath}")

print(f"\n{'='*60}")
print(f"ALL OUTPUTS SAVED TO: {LOG_DIR}")
print(f"  - {RUN_NAME}.log              (full console output)")
print(f"  - {RUN_NAME}_summary.json     (machine-readable results)")
print(f"  - hyperparams/                (saved hyperparameters)")
print(f"  - *.detections.*.txt          (detection results)")
print(f"{'='*60}")

# Close tee logger
end_time = datetime.now()
print(f"\nRun completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total wall time: {end_time - datetime.strptime(RUN_TIMESTAMP, '%Y%m%d_%H%M%S')}")
sys.stdout.flush()
sys.stderr.flush()

