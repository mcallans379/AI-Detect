#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Logistic Regression with SPOKEN LANGUAGE MARKERS to detect HumanSimulated AI.

NEW FEATURES targeting spoken transcripts vs AI-generated text:
  - filler_word_ratio: "um", "uh", "like", "you know", etc.
  - self_correction_ratio: "I mean", "wait", "actually", etc.
  - informal_contraction_ratio: "gonna", "wanna", "kinda", etc.
  - false_start_ratio: incomplete/abandoned phrases
  - repetition_ratio: consecutive word repetitions
  - discourse_marker_ratio: "so", "anyway", "basically", etc.

Hypothesis: Real transcripts from speech have MORE of these markers.
AI (even human-simulated) produces "too clean" text.
"""

import os
import re
import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

from joblib import dump

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

TRAIN_CSV = r"C:\projects\Spark\Live\Transcripts\2025_10_13\Transcripts_Identify_Cheaters\Train_Dec.csv"
HOLD_CSV  = r"C:\projects\Spark\Live\Transcripts\2025_10_13\Transcripts_Identify_Cheaters\Hold_Dec.csv"

TEXT_COL = "QA"
TYPE_COL = "Type"

OUTPUT_DIR  = r"C:\projects\Spark\AssessmentDevelopment\AI-research\models_logreg_binary_R_Dec"
MODEL_PATH  = os.path.join(OUTPUT_DIR, "logreg_binary_R_Dec.joblib")
VECT_PATH   = os.path.join(OUTPUT_DIR, "tfidf_R_Dec.joblib")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_R_Dec.joblib")
THRESH_PATH = os.path.join(OUTPUT_DIR, "threshold_R_Dec.json")

CROSSVAL_PRED_PATH = os.path.join(os.path.dirname(TRAIN_CSV), "Cross_Validation.csv")

# Target: Transcript FP rate must be <= 3%
TARGET_TRANSCRIPT_FP_RATE = 0.03

# Model settings
AI_CLASS_WEIGHT = 50
REGULARIZATION_C = 1.0

# Number of top features to show in rationale
TOP_N_FEATURES = 5

# ----------------------------------------------------------------------
# SPOKEN LANGUAGE PATTERNS (for detecting real transcripts)
# ----------------------------------------------------------------------

# Filler words common in spoken English
FILLER_WORDS = [
    'um', 'uh', 'uhm', 'umm', 'er', 'ah', 'eh',
    'like', 'you know', 'i mean', 'you see',
    'so', 'well', 'right', 'okay', 'ok',
    'basically', 'actually', 'literally', 'honestly',
    'kind of', 'sort of', 'i guess', 'i think',
]

# Self-correction phrases
SELF_CORRECTIONS = [
    'i mean', 'wait', 'no wait', 'actually', 'sorry',
    'let me', 'what i meant', 'i should say',
    'no no', 'well actually', 'or rather',
    'correction', 'i meant to say', 'that is',
]

# Informal contractions (spoken only, rarely written)
INFORMAL_CONTRACTIONS = [
    'gonna', 'wanna', 'gotta', 'kinda', 'sorta',
    'dunno', 'lemme', 'gimme', 'coulda', 'woulda', 'shoulda',
    'oughta', 'hafta', 'outta', 'lotsa', 'betcha',
    'gotcha', 'whatcha', 'howdy', "y'all", 'aint', "ain't",
    'til', "'til", 'cause', "'cause", 'cuz', "'cuz",
]

# Discourse markers (conversational transitions)
DISCOURSE_MARKERS = [
    'so', 'anyway', 'anyhow', 'besides', 'meanwhile',
    'however', 'therefore', 'moreover', 'furthermore',
    'basically', 'essentially', 'obviously', 'clearly',
    'right', 'okay', 'ok', 'alright', 'sure',
    'well', 'now', 'then', 'see', 'look',
]

# ----------------------------------------------------------------------
# TEXT STATISTICS FEATURES (25 total: 19 base + 6 spoken language)
# ----------------------------------------------------------------------

TEXT_STAT_FEATURE_NAMES = [
    # Base features (19 from Q_Dec)
    'log_char_count', 'log_word_count', 'log_sentence_count',
    'avg_sentence_length', 'std_sentence_length', 'min_sentence_length', 'max_sentence_length',
    'avg_word_length', 'std_word_length', 'long_word_ratio', 'short_word_ratio',
    'vocabulary_richness', 'hapax_ratio',
    'question_count', 'exclaim_count',
    'words_per_sentence', 'chars_per_word',
    'log_unique_words', 'sentence_length_range',
    # NEW: Spoken language markers (6 features)
    'filler_word_ratio',
    'self_correction_ratio', 
    'informal_contraction_ratio',
    'false_start_ratio',
    'repetition_ratio',
    'discourse_marker_ratio',
]


def count_pattern_occurrences(text_lower, patterns):
    """Count total occurrences of patterns in text."""
    count = 0
    for pattern in patterns:
        # Use word boundaries for single words, simple count for phrases
        if ' ' in pattern:
            count += text_lower.count(pattern)
        else:
            # Match whole words only
            count += len(re.findall(r'\b' + re.escape(pattern) + r'\b', text_lower))
    return count


def count_false_starts(text):
    """
    Count potential false starts / incomplete thoughts.
    Patterns: sentences ending with "...", dashes, or very short incomplete phrases.
    """
    count = 0
    
    # Ellipsis indicating trailing off
    count += len(re.findall(r'\.{2,}', text))
    
    # Dashes indicating interrupted thought
    count += len(re.findall(r'\s[-–—]\s', text))
    count += len(re.findall(r'[-–—]$', text))
    
    # Very short "sentences" (1-2 words) that might be false starts
    sentences = re.split(r'[.!?]+', text)
    for s in sentences:
        words = s.strip().split()
        if 1 <= len(words) <= 2 and not s.strip().endswith('?'):
            count += 1
    
    return count


def count_word_repetitions(text_lower):
    """
    Count consecutive word repetitions (e.g., "I I", "the the", "so so").
    These occur in natural speech when thinking.
    """
    words = text_lower.split()
    count = 0
    for i in range(len(words) - 1):
        if words[i] == words[i+1] and len(words[i]) > 1:
            count += 1
    return count


def compute_text_statistics(texts):
    """
    Compute statistical features from text.
    Includes base features + spoken language markers.
    """
    from collections import Counter
    features = []
    
    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            features.append([0] * 25)  # 25 features total
            continue
        
        text_lower = text.lower()
        
        char_count = len(text)
        word_list = text.split()
        word_count = len(word_list)
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)
        
        sentence_lengths = [len(s.split()) for s in sentences] if sentences else [0]
        avg_sentence_length = np.mean(sentence_lengths)
        std_sentence_length = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        min_sentence_length = np.min(sentence_lengths) if sentence_lengths else 0
        max_sentence_length = np.max(sentence_lengths) if sentence_lengths else 0
        
        word_lengths = [len(w) for w in word_list] if word_list else [0]
        avg_word_length = np.mean(word_lengths)
        std_word_length = np.std(word_lengths) if len(word_lengths) > 1 else 0
        
        long_word_ratio = sum(1 for w in word_list if len(w) > 8) / word_count if word_count > 0 else 0
        short_word_ratio = sum(1 for w in word_list if len(w) <= 3) / word_count if word_count > 0 else 0
        
        words_lower = [w.lower() for w in word_list]
        unique_words = len(set(words_lower))
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0
        
        word_freq = Counter(words_lower)
        hapax_ratio = sum(1 for w, c in word_freq.items() if c == 1) / unique_words if unique_words > 0 else 0
        
        question_count = text.count('?')
        exclaim_count = text.count('!')
        
        words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        chars_per_word = char_count / word_count if word_count > 0 else 0
        
        # =====================================================
        # NEW: SPOKEN LANGUAGE MARKERS
        # =====================================================
        
        # Filler words (real transcripts have more)
        filler_count = count_pattern_occurrences(text_lower, FILLER_WORDS)
        filler_word_ratio = filler_count / word_count if word_count > 0 else 0
        
        # Self-corrections (real transcripts have more)
        self_correction_count = count_pattern_occurrences(text_lower, SELF_CORRECTIONS)
        self_correction_ratio = self_correction_count / sentence_count if sentence_count > 0 else 0
        
        # Informal contractions (real transcripts have more)
        informal_count = count_pattern_occurrences(text_lower, INFORMAL_CONTRACTIONS)
        informal_contraction_ratio = informal_count / word_count if word_count > 0 else 0
        
        # False starts (real transcripts have more)
        false_start_count = count_false_starts(text)
        false_start_ratio = false_start_count / sentence_count if sentence_count > 0 else 0
        
        # Word repetitions (real transcripts have more)
        repetition_count = count_word_repetitions(text_lower)
        repetition_ratio = repetition_count / word_count if word_count > 0 else 0
        
        # Discourse markers (can go either way, but pattern may differ)
        discourse_count = count_pattern_occurrences(text_lower, DISCOURSE_MARKERS)
        discourse_marker_ratio = discourse_count / sentence_count if sentence_count > 0 else 0
        
        feat = [
            # Base features (19)
            np.log1p(char_count),
            np.log1p(word_count),
            np.log1p(sentence_count),
            avg_sentence_length,
            std_sentence_length,
            min_sentence_length,
            max_sentence_length,
            avg_word_length,
            std_word_length,
            long_word_ratio,
            short_word_ratio,
            vocabulary_richness,
            hapax_ratio,
            question_count,
            exclaim_count,
            words_per_sentence,
            chars_per_word,
            np.log1p(unique_words),
            max_sentence_length - min_sentence_length,
            # Spoken language markers (6)
            filler_word_ratio,
            self_correction_ratio,
            informal_contraction_ratio,
            false_start_ratio,
            repetition_ratio,
            discourse_marker_ratio,
        ]
        
        features.append(feat)
    
    return np.array(features)


# ----------------------------------------------------------------------
# EXPLAINABILITY FUNCTIONS
# ----------------------------------------------------------------------

def get_feature_contributions(X_row, coefficients, feature_names):
    """Calculate the contribution of each feature to the prediction."""
    if hasattr(X_row, 'toarray'):
        X_row = X_row.toarray().flatten()
    else:
        X_row = np.array(X_row).flatten()
    
    contributions = []
    for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
        if i < len(X_row):
            contrib = X_row[i] * coef
            if contrib != 0:
                contributions.append((name, contrib))
    
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions


def generate_rationale(X_row, coefficients, feature_names, pred_label, top_n=5):
    """Generate a human-readable rationale for the prediction."""
    contributions = get_feature_contributions(X_row, coefficients, feature_names)
    
    if pred_label == 1:
        ai_features = [(name, contrib) for name, contrib in contributions if contrib > 0][:top_n]
        if ai_features:
            parts = [f"{name}(+{contrib:.3f})" for name, contrib in ai_features]
            return "AI indicators: " + ", ".join(parts)
        else:
            return "AI (weak signal)"
    else:
        trans_features = [(name, contrib) for name, contrib in contributions if contrib < 0][:top_n]
        if trans_features:
            parts = [f"{name}({contrib:.3f})" for name, contrib in trans_features]
            return "Transcript indicators: " + ", ".join(parts)
        else:
            return "Transcript (weak signal)"


def generate_detailed_rationale(X_row, coefficients, feature_names, prob_ai, threshold, top_n=5):
    """Generate detailed rationale showing both AI and Transcript indicators."""
    contributions = get_feature_contributions(X_row, coefficients, feature_names)
    
    ai_features = [(name, contrib) for name, contrib in contributions if contrib > 0][:top_n]
    trans_features = [(name, contrib) for name, contrib in contributions if contrib < 0][:top_n]
    
    parts = []
    
    if ai_features:
        ai_parts = [f"{name}(+{contrib:.2f})" for name, contrib in ai_features]
        parts.append("AI: " + ", ".join(ai_parts))
    
    if trans_features:
        trans_parts = [f"{name}({contrib:.2f})" for name, contrib in trans_features]
        parts.append("Trans: " + ", ".join(trans_parts))
    
    return " | ".join(parts) if parts else "No strong indicators"


# ----------------------------------------------------------------------
# OTHER FUNCTIONS
# ----------------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path, encoding="latin1")
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def map_labels(df, type_col=TYPE_COL):
    types_norm = df[type_col].astype(str).str.strip().str.lower()
    is_transcript = (types_norm == "transcript")
    y = (~is_transcript).astype(int).values
    print(f"  Transcript: {(y==0).sum()}, AI: {(y==1).sum()}")
    return y


def build_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        min_df=2,
        max_df=0.95,
        lowercase=True,
        sublinear_tf=True,
    )


def find_threshold_for_fp_rate(y_true, y_proba, target_fp_rate):
    """Find threshold to keep Transcript FP rate at or below target."""
    mask_trans = (y_true == 0)
    proba_trans = y_proba[mask_trans]
    n_trans = len(proba_trans)
    n_ai = (y_true == 1).sum()
    
    if n_trans == 0:
        return 0.5, {"threshold": 0.5, "trans_fp": 0, "trans_fp_rate": 0.0,
                     "ai_tp": 0, "ai_fn": 0, "ai_recall": 0.0, "n_trans": 0, "n_ai": int(n_ai)}
    
    sorted_trans_probs = np.sort(proba_trans)[::-1]
    n_to_flag = max(1, int(n_trans * target_fp_rate))
    
    if n_to_flag >= len(sorted_trans_probs):
        threshold = sorted_trans_probs[-1] - 0.0001
    else:
        threshold = sorted_trans_probs[n_to_flag - 1] - 0.0001
    
    y_pred = (y_proba >= threshold).astype(int)
    
    trans_fp = ((y_true == 0) & (y_pred == 1)).sum()
    trans_fp_rate = trans_fp / n_trans
    
    ai_tp = ((y_true == 1) & (y_pred == 1)).sum()
    ai_fn = ((y_true == 1) & (y_pred == 0)).sum()
    ai_recall = ai_tp / (ai_tp + ai_fn) if (ai_tp + ai_fn) > 0 else 0
    
    return threshold, {
        "threshold": float(threshold),
        "trans_fp": int(trans_fp),
        "trans_fp_rate": float(trans_fp_rate),
        "ai_tp": int(ai_tp),
        "ai_fn": int(ai_fn),
        "ai_recall": float(ai_recall),
        "n_trans": int(n_trans),
        "n_ai": int(n_ai),
    }


def show_tradeoffs(y_true, y_proba):
    """Show AI recall at different Transcript FP rates"""
    print("\n" + "="*60)
    print("TRADEOFF: Transcript FP Rate vs AI Recall")
    print("="*60)
    print(f"{'FP Rate':>10} {'Trans FP':>10} {'AI Recall':>12} {'AI TP':>8} {'AI FN':>8}")
    print("-"*60)
    
    for fp_rate in [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]:
        _, m = find_threshold_for_fp_rate(y_true, y_proba, fp_rate)
        print(f"{fp_rate*100:>9.1f}% {m['trans_fp']:>10d} {m['ai_recall']*100:>11.1f}% "
              f"{m['ai_tp']:>8d} {m['ai_fn']:>8d}")
    print("-"*60)


def show_breakdown_by_type(df, y_pred, type_col=TYPE_COL):
    """Show detection rate for each Type"""
    print("\n" + "="*60)
    print("DETECTION BY TYPE")
    print("="*60)
    
    df = df.copy()
    df["pred"] = y_pred
    
    for type_val in sorted(df[type_col].unique()):
        mask = df[type_col] == type_val
        subset = df[mask]
        detected = (subset["pred"] == 1).sum()
        total = len(subset)
        pct = detected/total*100 if total > 0 else 0
        bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
        print(f"  {type_val:25s} {detected:3d}/{total:3d} ({pct:5.1f}%) {bar}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("LOGISTIC REGRESSION WITH SPOKEN LANGUAGE MARKERS")
    print("Targeting HumanSimulated AI detection")
    print(f"{'='*60}")
    print(f"AI Class Weight: {AI_CLASS_WEIGHT}")
    print(f"Target Transcript FP Rate: <= {TARGET_TRANSCRIPT_FP_RATE*100:.1f}%")
    print(f"Text stat features: {len(TEXT_STAT_FEATURE_NAMES)} (19 base + 6 spoken)")
    print(f"{'='*60}\n")

    # Load data
    train_df = load_data(TRAIN_CSV)
    hold_df = load_data(HOLD_CSV)
    
    train_df[TEXT_COL] = train_df[TEXT_COL].fillna("")
    hold_df[TEXT_COL] = hold_df[TEXT_COL].fillna("")
    
    y_train = map_labels(train_df)
    y_hold = map_labels(hold_df)

    # TF-IDF Features
    print("\nBuilding TF-IDF features...")
    vect = build_vectorizer()
    X_train_tfidf = vect.fit_transform(train_df[TEXT_COL])
    X_hold_tfidf = vect.transform(hold_df[TEXT_COL])
    print(f"  TF-IDF shape: {X_train_tfidf.shape}")

    # Text Statistics Features (with spoken language markers)
    print("\nComputing text statistics features (with spoken language markers)...")
    X_train_stats = compute_text_statistics(train_df[TEXT_COL].values)
    X_hold_stats = compute_text_statistics(hold_df[TEXT_COL].values)
    print(f"  Text stats shape: {X_train_stats.shape}")
    
    # Show spoken language marker statistics
    print("\n  Spoken Language Marker Averages (Train):")
    spoken_features = ['filler_word_ratio', 'self_correction_ratio', 'informal_contraction_ratio',
                       'false_start_ratio', 'repetition_ratio', 'discourse_marker_ratio']
    spoken_indices = [TEXT_STAT_FEATURE_NAMES.index(f) for f in spoken_features]
    
    for idx, name in zip(spoken_indices, spoken_features):
        trans_avg = X_train_stats[y_train == 0, idx].mean()
        ai_avg = X_train_stats[y_train == 1, idx].mean()
        diff = trans_avg - ai_avg
        direction = "↑ Transcript" if diff > 0 else "↑ AI"
        print(f"    {name:30s}: Trans={trans_avg:.4f}, AI={ai_avg:.4f} ({direction})")
    
    # Scale the statistics features
    scaler = StandardScaler()
    X_train_stats_scaled = scaler.fit_transform(X_train_stats)
    X_hold_stats_scaled = scaler.transform(X_hold_stats)

    # Combine TF-IDF + Statistics
    print("\nCombining features...")
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_stats_scaled)])
    X_hold = hstack([X_hold_tfidf, csr_matrix(X_hold_stats_scaled)])
    print(f"  Combined shape: {X_train.shape}")

    # Get all feature names
    tfidf_feature_names = list(vect.get_feature_names_out())
    all_feature_names = tfidf_feature_names + TEXT_STAT_FEATURE_NAMES
    print(f"  Total features: {len(all_feature_names):,}")

    # Train final model
    print(f"\nTraining logistic regression (class_weight={{0:1, 1:{AI_CLASS_WEIGHT}}})...")
    final_model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=REGULARIZATION_C,
        class_weight={0: 1, 1: AI_CLASS_WEIGHT}
    )
    final_model.fit(X_train, y_train)
    
    coefficients = final_model.coef_[0]
    
    # Show coefficients for spoken language features
    print("\n" + "="*60)
    print("SPOKEN LANGUAGE FEATURE COEFFICIENTS")
    print("(Negative = indicates Transcript, Positive = indicates AI)")
    print("="*60)
    for name in spoken_features:
        idx = len(tfidf_feature_names) + TEXT_STAT_FEATURE_NAMES.index(name)
        coef = coefficients[idx]
        direction = "→ AI" if coef > 0 else "→ Transcript"
        print(f"  {name:30s}: {coef:+.4f} ({direction})")

    # HOLD SET EVALUATION
    print("\n" + "="*60)
    print("HOLD SET EVALUATION")
    print("="*60)
    
    proba_hold = final_model.predict_proba(X_hold)[:, 1]
    
    # Show tradeoffs
    show_tradeoffs(y_hold, proba_hold)
    
    # Get threshold for target FP rate
    best_thresh, metrics = find_threshold_for_fp_rate(y_hold, proba_hold, TARGET_TRANSCRIPT_FP_RATE)
    
    print(f"\nAt {TARGET_TRANSCRIPT_FP_RATE*100:.1f}% FP target:")
    print(f"  Threshold: {best_thresh:.4f}")
    print(f"  Transcript FP: {metrics['trans_fp']} ({metrics['trans_fp_rate']*100:.2f}%)")
    print(f"  AI Recall: {metrics['ai_recall']*100:.2f}%")
    print(f"  AI TP: {metrics['ai_tp']}, FN: {metrics['ai_fn']}")

    y_pred = (proba_hold >= best_thresh).astype(int)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_hold, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_hold, y_pred, digits=3))
    
    # Breakdown by type
    show_breakdown_by_type(hold_df, y_pred)

    # Generate rationales for each prediction
    print("\nGenerating rationales for each prediction...")
    rationales = []
    detailed_rationales = []
    
    for i in range(X_hold.shape[0]):
        X_row = X_hold[i]
        pred = y_pred[i]
        prob = proba_hold[i]
        
        rationale = generate_rationale(X_row, coefficients, all_feature_names, pred, TOP_N_FEATURES)
        detailed = generate_detailed_rationale(X_row, coefficients, all_feature_names, prob, best_thresh, TOP_N_FEATURES)
        
        rationales.append(rationale)
        detailed_rationales.append(detailed)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{X_hold.shape[0]} rows...")

    # Save HOLD predictions with rationales to Cross_Validation.csv
    hold_out = hold_df.copy()
    hold_out["prob_ai"] = proba_hold
    hold_out["pred_label"] = y_pred
    hold_out["pred_class_name"] = np.where(y_pred == 1, "AI", "Transcript")
    hold_out["true_label"] = y_hold
    hold_out["is_correct"] = (y_pred == y_hold).astype(int)
    hold_out["rationale"] = rationales
    hold_out["detailed_rationale"] = detailed_rationales
    
    cols = ["who", "personID", TEXT_COL, TYPE_COL, "true_label", "prob_ai", 
            "pred_label", "pred_class_name", "is_correct", "rationale", "detailed_rationale"]
    hold_out = hold_out[[c for c in cols if c in hold_out.columns]]
    hold_out.to_csv(CROSSVAL_PRED_PATH, index=False)
    print(f"\nSaved HOLD predictions with rationales to {CROSSVAL_PRED_PATH}")

    # Save model artifacts
    dump(final_model, MODEL_PATH)
    dump(vect, VECT_PATH)
    dump(scaler, SCALER_PATH)
    
    # Export for PHP
    print("\n" + "="*60)
    print("EXPORTING MODEL FOR PHP")
    print("="*60)
    
    vocabulary = {k: int(v) for k, v in vect.vocabulary_.items()}
    idf_values = [float(x) for x in vect.idf_]
    coefficients_list = [float(x) for x in coefficients]
    intercept = float(final_model.intercept_[0])
    scaler_mean = [float(x) for x in scaler.mean_]
    scaler_scale = [float(x) for x in scaler.scale_]
    
    php_model = {
        "vocabulary": vocabulary,
        "idf_values": idf_values,
        "coefficients": coefficients_list,
        "intercept": intercept,
        "threshold": float(best_thresh),
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "text_stat_feature_names": TEXT_STAT_FEATURE_NAMES,
        "n_tfidf_features": len(vocabulary),
        "n_stat_features": len(TEXT_STAT_FEATURE_NAMES),
        "ngram_range": [1, 3],
        "target_fp_rate": TARGET_TRANSCRIPT_FP_RATE,
        # Include spoken language patterns for PHP
        "filler_words": FILLER_WORDS,
        "self_corrections": SELF_CORRECTIONS,
        "informal_contractions": INFORMAL_CONTRACTIONS,
        "discourse_markers": DISCOURSE_MARKERS,
    }
    
    php_model_path = os.path.join(OUTPUT_DIR, "model_for_php.json")
    with open(php_model_path, "w", encoding="utf-8") as f:
        json.dump(php_model, f)
    
    print(f"Saved PHP model to {php_model_path}")

    # Save feature CSVs
    features_df = pd.DataFrame({
        'feature': all_feature_names,
        'coefficient': coefficients_list,
        'abs_coefficient': np.abs(coefficients_list),
        'feature_type': ['tfidf'] * len(tfidf_feature_names) + ['text_stat'] * len(TEXT_STAT_FEATURE_NAMES)
    })
    features_df = features_df.sort_values('abs_coefficient', ascending=False)
    
    features_df.to_csv(os.path.join(OUTPUT_DIR, "all_features.csv"), index=False)
    features_df[features_df['coefficient'] > 0].to_csv(os.path.join(OUTPUT_DIR, "features_for_AI.csv"), index=False)
    features_df[features_df['coefficient'] < 0].to_csv(os.path.join(OUTPUT_DIR, "features_for_Transcript.csv"), index=False)
    features_df[features_df['feature_type'] == 'text_stat'].to_csv(os.path.join(OUTPUT_DIR, "text_statistics_features.csv"), index=False)

    with open(THRESH_PATH, "w") as f:
        json.dump({
            "threshold": float(best_thresh),
            "target_fp_rate": float(TARGET_TRANSCRIPT_FP_RATE),
            **{k: (int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else 
                   float(v) if isinstance(v, (np.floating, np.float64)) else v)
               for k, v in metrics.items()}
        }, f, indent=2)
    
    print(f"\nSaved all artifacts to {OUTPUT_DIR}")
    print("\n" + "="*60)
    print("NEW SPOKEN LANGUAGE FEATURES:")
    print("  - filler_word_ratio (um, uh, like, you know...)")
    print("  - self_correction_ratio (I mean, wait, actually...)")
    print("  - informal_contraction_ratio (gonna, wanna, kinda...)")
    print("  - false_start_ratio (incomplete thoughts, ...)")
    print("  - repetition_ratio (consecutive word repeats)")
    print("  - discourse_marker_ratio (so, anyway, basically...)")
    print("="*60)
    print("\nDone!")


if __name__ == "__main__":
    main()


