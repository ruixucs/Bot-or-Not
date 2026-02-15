
import json
import re
import os
from collections import defaultdict, Counter

# ============================================================
# Configuration
# ============================================================
DATASET_FILES = {
    "en": "dataset.posts&users.34.json",
    "fr": "dataset.posts&users.35.json",
}

DETECTION_DIR = os.path.join("runs", "max_v3_20260214_131317")
DETECTION_FILES = {
    "en": os.path.join(DETECTION_DIR, "Bot_killer_RX.detections.en.txt"),
    "fr": os.path.join(DETECTION_DIR, "Bot_killer_RX.detections.fr.txt"),
    "all": os.path.join(DETECTION_DIR, "Bot_killer_RX.detections.all.txt"),
}

# ============================================================
# Engagement Farming Patterns
# ============================================================

# --- English: solicitation in post text ---
EN_SOLICITATION = [
    # Ask for retweets
    r'\blike\s+and\s+retweet\b',
    r'\bretweet\s+and\s+(?:like|follow|share)\b',
    r'\b(?:please|kindly|pls)\s+retweet\b',
    r'\bretweet\s+(?:this|my|our|if)\b',
    r'\b(?:pls|plz)\s+rt\b',
    # Ask for follows
    r'\bmust\s+follow\b',
    r'\bfollow\s+(?:me|us)\s+(?:for|to|and|on|if)\b',
    r'\bfollow\s+(?:4|for)\s+follow\b',
    r'\bf4f\b',
    # Ask for likes
    r'\blike\s+(?:4|for)\s+like\b',
    r'\bl4l\b',
    # Spread / share
    r'\bspread\s+the\s+word\b',
    r'\bshare\s+(?:this|and|&)\b.*\b(?:retweet|follow|like)\b',
    # Combined "like, retweet and follow" patterns
    r'\b(?:like|retweet|rt)\s*[,&]\s*(?:like|retweet|rt|follow)\b',
]

# --- French: solicitation in post text ---
FR_SOLICITATION = [
    # "N'hésitez pas à RT / retweeter / liker / partager"
    r"n['\u2019]h[ée]site[zs]?\s+pas\s+[àa]\s+(?:RT|retweete[rz]|like[rz]|partage[rz])",
    # Direct retweet ask: "retweetez", "retweeter svp"
    r'\bretweete[zr]\b',
    # "RT et follow", "RT svp", "RT stp"
    r'\bRT\s+(?:svp|stp|s\.?v\.?p)\b',
    r'\bRT\s+et\s+follow\b',
    # Follow solicitation: "follow nous/moi/me", "me/nous follow"
    r'\bfollow\s+(?:nous|moi)\b',
    r'\b(?:nous|moi|me)\s+follow\b',
    # "abonnez-vous", "abonne-toi"
    r'\babonne[zr]?\s*[-\s]?\s*(?:vous|toi)\b',
    # Daily repetitive: "Jour 37 pour que mon club me follow"
    r'\bjour\s+\d+\s+pour\s+que\b',
    # "liker, RT et (nous) follow"
    r'\blike[rz]?\s*,?\s*(?:RT|retweet)\s+et\s+(?:nous\s+)?follow\b',
    # "like et follow", "like et RT"
    r'\blike[rz]?\s+et\s+(?:follow|RT|retweet)\b',
    # "pour la force" (common engagement farming closer)
    r'\bpour\s+la\s+force\b',
]

# --- Bio patterns (both languages) ---
BIO_SOLICITATION = [
    r'\bfollow\s+me\s+(?:i|I)\s+follow\s*back\b',
    r'\bi\s+follow\s+back\b',
    r'\bfollow\s+back\b',
    r'\bf4f\b',
    r'\bfollow\s+(?:4|for)\s+follow\b',
    r'\bje\s+follow\s+back\b',
    r'\bfollow\s+me\b.*\bfollow\b',
]



# ============================================================
# Hex / Encoding Anomaly Patterns
# ============================================================

# Common French accented char hex codes (Latin-1 / ISO-8859-1):
#   e9=é  e8=è  e0=à  f4=ô  e7=ç  f9=ù  c3=Ã (UTF-8 lead byte)
#
# NOTE: Purely-alphabetic codes are EXCLUDED to avoid massive false positives:
#   ea=ê  ee=î  ef=ï  eb=ë  fb=û
# These match extremely common words (e.g. "great"→ea, "need"→ee,
# "refer"→ef, "rebel"→eb, "offbeat"→fb).  The digit-containing codes
# below are unambiguous because digits never appear inside normal words.
HEX_ACCENT_CODES = r'(?:e9|e8|e0|f4|e7|f9|c3)'
# Word containing hex accent leak: [letter(s)] + hex_code + [letter(s)]
# e.g. "cine9ma" (cinéma), "e9pique" (épique), "the8me" (thème)
HEX_IN_WORD_RE = re.compile(
    r'[a-zA-ZÀ-ÿ]' + HEX_ACCENT_CODES + r'[a-zA-ZÀ-ÿ]',
    re.IGNORECASE
)

# Raw hex emoji/unicode code points in text (e.g. "01f627a0", "01f60d")
# 6-8 hex chars that look like unicode code points, not normal words
HEX_EMOJI_RE = re.compile(
    r'\b0[0-9a-f]{4,7}\b',  # starts with 0, like U+01F627
    re.IGNORECASE
)

# Standalone hex sequences (4+ hex chars not forming common words)
# More aggressive: catches things like "a0" in isolation, hex dumps
HEX_STANDALONE_RE = re.compile(
    r'\b[0-9a-f]{6,}\b',  # 6+ hex chars in a row
    re.IGNORECASE
)

# JSON array format leak: ['text'] or ["text"]
JSON_LEAK_RE = re.compile(
    r"""\[['"].*?['"]\]""",
    re.DOTALL
)

# Words composed entirely of hex characters (a-f, 0-9) that should NOT
# trigger encoding-anomaly detection via HEX_STANDALONE_RE (6+ chars).
# Without this filter, normal English words like "beefed" or "decade"
# would be flagged as hex encoding leaks.
COMMON_HEX_WORDS = {
    # 6-letter (minimum length matched by HEX_STANDALONE_RE)
    'accede', 'beaded', 'beefed', 'bedded', 'dabbed', 'decade',
    'deface', 'deeded', 'efface', 'facade', 'faffed', 'fagged',
    # 7-letter
    'acceded', 'defaced', 'effaced',
    # test / placeholder strings
    'abcdef', 'aabbcc', 'ddeeff',
}


def compile_patterns(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]


EN_COMPILED = compile_patterns(EN_SOLICITATION)
FR_COMPILED = compile_patterns(FR_SOLICITATION)
BIO_COMPILED = compile_patterns(BIO_SOLICITATION)


# ============================================================
# Core Detection Logic
# ============================================================

def load_dataset(filepath):
    """Load dataset JSON -> (users_dict, posts_by_author)."""
    print(f"  Loading {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    users = {u['id']: u for u in data['users']}
    posts_by_user = defaultdict(list)
    for post in data['posts']:
        posts_by_user[post['author_id']].append(post)
    print(f"  -> {len(users)} users, {sum(len(v) for v in posts_by_user.values())} posts")
    return users, posts_by_user


def count_solicitation_posts(posts, patterns):
    """Count posts matching solicitation patterns."""
    count = 0
    samples = []
    for post in posts:
        text = post.get('text', '')
        for pat in patterns:
            if pat.search(text):
                count += 1
                samples.append(text[:120].replace('\n', ' '))
                break
    return count, samples


def check_bio(user):
    """Check if user bio matches solicitation patterns."""
    desc = (user.get('description') or '')
    for pat in BIO_COMPILED:
        if pat.search(desc):
            return True
    return False


def detect_repetitive_template(posts):
    """Detect daily repetitive patterns (e.g. 'Jour 37 pour que...', 'Jour 38 pour que...')."""
    if len(posts) < 3:
        return 0
    normalized = []
    for post in posts:
        text = post.get('text', '').strip()
        norm = re.sub(r'\d+', '<N>', text)
        normalized.append(norm)
    counts = Counter(normalized)
    return max(counts.values()) if counts else 0


def count_hex_accent_posts(posts):
    """
    Count posts containing hex accent leaks (e.g. 'cine9ma' for 'cinéma').
    Returns (count_of_posts_with_hex, total_hex_hits, sample_texts).
    """
    post_count = 0
    total_hits = 0
    samples = []
    for post in posts:
        text = post.get('text', '')
        matches = HEX_IN_WORD_RE.findall(text)
        if matches:
            post_count += 1
            total_hits += len(matches)
            # Show the hex fragments found
            preview = text[:100].replace('\n', ' ')
            samples.append(f"{preview}  [hex: {', '.join(matches[:5])}]")
    return post_count, total_hits, samples


def count_hex_emoji_posts(posts):
    """
    Count posts containing raw hex emoji/unicode code points
    (e.g. '01f627a0', '01f60d').
    """
    post_count = 0
    samples = []
    for post in posts:
        text = post.get('text', '')
        # Look for emoji-like hex codes AND standalone hex sequences
        emoji_matches = HEX_EMOJI_RE.findall(text)
        standalone_matches = HEX_STANDALONE_RE.findall(text)
        all_matches = list(set(emoji_matches + standalone_matches))
        # Filter out:
        #  1) common English words composed entirely of hex chars
        #  2) pure numbers (no a-f letters) — these are dates/counts, not hex leaks
        #     e.g. "13500000", "032824" (=03/28/24), "5000000"
        filtered = [m for m in all_matches
                    if m.lower() not in COMMON_HEX_WORDS
                    and re.search(r'[a-fA-F]', m)]
        if filtered:
            post_count += 1
            preview = text[:100].replace('\n', ' ')
            samples.append(f"{preview}  [hex: {', '.join(filtered[:3])}]")
    return post_count, samples


def count_json_leak_posts(posts):
    """
    Count posts containing JSON array format leak: ['text'] or ["text"].
    This is a pipeline artifact where raw JSON was not unpacked.
    """
    count = 0
    samples = []
    for post in posts:
        text = post.get('text', '')
        if JSON_LEAK_RE.search(text):
            count += 1
            samples.append(text[:120].replace('\n', ' '))
    return count, samples


def detect_bots(lang, dataset_file):
    """Scan a dataset for engagement farming bots AND hex/encoding anomaly bots."""
    users, posts_by_user = load_dataset(dataset_file)
    patterns = EN_COMPILED if lang == 'en' else FR_COMPILED

    detected = []

    for uid, user in users.items():
        user_posts = posts_by_user.get(uid, [])
        total = len(user_posts)
        if total == 0:
            continue

        # --- Engagement Farming signals ---
        sol_count, sol_samples = count_solicitation_posts(user_posts, patterns)
        sol_ratio = sol_count / total if total > 0 else 0
        bio_match = check_bio(user)
        max_repeat = detect_repetitive_template(user_posts)

        # --- Hex / Encoding signals ---
        hex_accent_posts, hex_accent_hits, hex_accent_samples = count_hex_accent_posts(user_posts)
        hex_emoji_posts, hex_emoji_samples = count_hex_emoji_posts(user_posts)
        json_leak_posts, json_leak_samples = count_json_leak_posts(user_posts)

        # ---------- Decision Rules ----------
        is_bot = False
        reason = ""
        samples = []

        # ===== Engagement Farming Rules =====

        # R-EF1: High solicitation count (>=4 posts)
        if sol_count >= 4:
            is_bot = True
            reason = f"R-EF1 HIGH_SOLICITATION: {sol_count}/{total} posts ({sol_ratio:.0%})"
            samples = sol_samples

        # R-EF2: Moderate solicitation + meaningful ratio (>=2 posts, >=10%)
        elif sol_count >= 2 and sol_ratio >= 0.10:
            is_bot = True
            reason = f"R-EF2 MOD_SOLICITATION: {sol_count}/{total} posts ({sol_ratio:.0%})"
            samples = sol_samples

        # R-EF3: Bio match + any solicitation in posts
        elif bio_match and sol_count >= 1:
            is_bot = True
            reason = f"R-EF3 BIO+POSTS: bio='follow back' + {sol_count} solicitation posts"
            samples = sol_samples

        # R-EF4: Bio match + very low post count (likely fake/engagement account)
        elif bio_match and total <= 12:
            is_bot = True
            reason = f"R-EF4 BIO+LOW_ACTIVITY: bio='follow back', only {total} posts"
            samples = sol_samples

        # R-EF5: Repetitive template + solicitation (e.g. "Jour X pour que...")
        elif max_repeat >= 3 and sol_count >= 1:
            is_bot = True
            reason = f"R-EF5 REPETITIVE: {max_repeat} near-dup posts + {sol_count} solicitation"
            samples = sol_samples

        # ===== Hex / Encoding Anomaly Rules =====

        # R-HEX1: Multiple posts with hex accent leaks (LLM encoding artifact)
        #   e.g. "cine9ma"(=cinéma), "e9pique"(=épique), "de9lire"(=délire)
        #   >=2 posts with hex accent OR >=3 total hex hits in any posts
        elif hex_accent_posts >= 2 or hex_accent_hits >= 3:
            is_bot = True
            reason = (f"R-HEX1 HEX_ACCENT_LEAK: {hex_accent_posts}/{total} posts, "
                      f"{hex_accent_hits} total hex fragments")
            samples = hex_accent_samples

        # R-HEX2: Posts containing raw hex emoji/unicode code points
        #   e.g. "01f627a0" appearing as literal text
        #   Require >=2 posts to avoid false positives from single hex-like tokens
        elif hex_emoji_posts >= 2:
            is_bot = True
            reason = f"R-HEX2 HEX_EMOJI_LEAK: {hex_emoji_posts}/{total} posts with raw hex codes"
            samples = hex_emoji_samples

        # R-JSON: Posts containing JSON array format leak ['text'] or ["text"]
        #   Pipeline artifact where raw JSON wasn't unpacked
        elif json_leak_posts >= 1:
            is_bot = True
            reason = f"R-JSON JSON_ARRAY_LEAK: {json_leak_posts}/{total} posts with ['...'] format"
            samples = json_leak_samples

        if is_bot:
            detected.append({
                'uid': uid,
                'username': user.get('username', '?'),
                'description': (user.get('description') or '')[:80],
                'reason': reason,
                'sol_count': sol_count,
                'hex_accent_hits': hex_accent_hits,
                'json_leak_posts': json_leak_posts,
                'total': total,
                'bio_match': bio_match,
                'samples': samples[:3],
            })

    return detected


# ============================================================
# File I/O
# ============================================================

def load_ids(filepath):
    """Load a detection file, return ordered list of user IDs."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def save_ids(filepath, ids):
    """Write detection file: one user ID per line."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for uid in ids:
            f.write(f"{uid}\n")


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("=" * 65)
    print("  POST-PROCESSING BOT DETECTOR  (DRY RUN — print only)")
    print("  Engagement Farming + Hex/Encoding Anomalies")
    print("  Target: Bot_killer_RX detections")
    print("=" * 65)

    all_new_ids = []

    for lang in ['en', 'fr']:
        ds_file = DATASET_FILES[lang]
        det_file = DETECTION_FILES[lang]

        print(f"\n{'─' * 65}")
        print(f"  [{lang.upper()}] Scanning {ds_file}")
        print(f"{'─' * 65}")

        # Detect engagement farmers + hex/encoding anomaly bots
        detected = detect_bots(lang, ds_file)

        # Compare with existing detections
        existing_ids = set(load_ids(det_file))
        new_bots = [d for d in detected if d['uid'] not in existing_ids]
        already  = [d for d in detected if d['uid'] in existing_ids]

        # Categorize detections by rule type
        ef_count = sum(1 for d in detected if d['reason'].startswith('R-EF'))
        hex_count = sum(1 for d in detected if d['reason'].startswith('R-HEX'))
        json_count = sum(1 for d in detected if d['reason'].startswith('R-JSON'))

        print(f"\n  Summary:")
        print(f"    Existing detections in file:  {len(existing_ids)}")
        print(f"    Total bots found this scan:   {len(detected)}")
        print(f"      Engagement Farming (R-EF):  {ef_count}")
        print(f"      Hex Encoding Leak (R-HEX):  {hex_count}")
        print(f"      JSON Array Leak (R-JSON):   {json_count}")
        print(f"      -> Already in detections:   {len(already)}")
        print(f"      -> NEW (not yet in file):   {len(new_bots)}")

        if already:
            print(f"\n  [Already caught — no action needed]")
            for d in already:
                print(f"    OK  {d['uid'][:12]}...  @{d['username']}")
                print(f"        {d['reason']}")

        if new_bots:
            print(f"\n  [NEW detections for {lang.upper()}]")
            for d in new_bots:
                print(f"    +++ {d['uid']}  @{d['username']}")
                print(f"        Reason: {d['reason']}")
                if d['bio_match']:
                    print(f"        Bio:    \"{d['description']}\"")
                for s in d['samples']:
                    print(f"        Sample: \"{s}\"")
                print()

            all_new_ids.extend(d['uid'] for d in new_bots)
        else:
            print(f"\n  => No new bots found for {lang.upper()}.")

    # --- Final summary (print only, no file writes) ---
    print(f"\n{'=' * 65}")
    print(f"  DRY RUN COMPLETE")
    print(f"  Total new bots found: {len(all_new_ids)}")
    if all_new_ids:
        print(f"\n  New IDs (not written to file):")
        for uid in all_new_ids:
            print(f"    {uid}")
    else:
        print(f"  No new bots to add.")
    print(f"{'=' * 65}")
    print()


if __name__ == "__main__":
    main()
