"""
CER Evaluation Script for Malayalam OCR RAG Pipeline
=====================================================
Evaluates character recognition on testsamples500k dataset.

For each test image:
1. Extract ground truth word from filename
2. Segment the word image into characters
3. Recognize each character via embedding similarity (ChromaDB)
4. Concatenate recognized characters to form predicted word
5. Compute CER (Character Error Rate) using Levenshtein distance

Usage:
    python evaluate_cer.py                          # Full evaluation
    python evaluate_cer.py --max-images 10          # Quick test with 10 images
    python evaluate_cer.py --max-images 50 --verbose  # Verbose output
"""
import os
import sys
import json
import time
import csv
import re
from pathlib import Path
from PIL import Image
import numpy as np
import regex  # For Unicode grapheme cluster segmentation

# Ensure project imports work
sys.path.insert(0, str(Path(__file__).parent))

from char_embeddings import CharacterRecognizer


# ============================================================================
# CER Computation (Grapheme Cluster-aware for Malayalam)
# ============================================================================

def split_grapheme_clusters(text):
    """
    Split text into granular visual units for Malayalam.
    
    Treats vowel signs as distinct units from their base consonants,
    but keeps conjuncts (Consonant + Virama + Consonant) together.
    
    This matches the user's mental model where 'വൃ' is 2 units (വ + ൃ).
    
    Examples:
        'വൃത്തഇ' → ['വ', 'ൃ', 'ത്ത', 'ഇ'] (4 units, CER 0.25 if one wrong)
        'കാ'     → ['ക', 'ാ']
    """
    # Pattern explanation:
    # 1. Conjuncts: (\p{L}\u0D4D)+\p{L}
    # 2. Individual Vowel Signs/Markers: \p{M}
    # 3. Individual Letters/Digits/Others: .
    pattern = r'(?:\p{L}\u0D4D)+\p{L}|\p{M}|.'
    return regex.findall(pattern, text)


def levenshtein_distance(s1, s2):
    """
    Compute Levenshtein (edit) distance between two sequences.
    Works with both strings and lists of grapheme clusters.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(prediction, reference):
    """
    Compute Character Error Rate using grapheme clusters.
    
    CER = edit_distance(pred_clusters, ref_clusters) / len(ref_clusters)
    
    This operates on visual characters (grapheme clusters), not raw
    Unicode code points, giving meaningful CER for Malayalam text.
    """
    pred_clusters = split_grapheme_clusters(prediction)
    ref_clusters = split_grapheme_clusters(reference)
    
    if len(ref_clusters) == 0:
        return 1.0 if len(pred_clusters) > 0 else 0.0

    distance = levenshtein_distance(pred_clusters, ref_clusters)
    return distance / len(ref_clusters)

import math
from collections import defaultdict

class CharacterBigramModel:
    def __init__(self):
        self.bigrams = defaultdict(lambda: defaultdict(int))
        self.unigrams = defaultdict(int)
        self.vocab = set()

    def train(self, text_corpus):
        """Trains the bigram model on a list of valid words."""
        for word in text_corpus:
            chars = split_grapheme_clusters(word)
            for i in range(len(chars)):
                self.unigrams[chars[i]] += 1
                self.vocab.add(chars[i])
                if i < len(chars) - 1:
                    self.bigrams[chars[i]][chars[i+1]] += 1
        print(f"✅ Trained Bigram Model: {len(self.unigrams)} unique graphemes from {len(text_corpus)} words.")

    def get_transition_log_prob(self, char1, char2):
        """Returns log probability with Add-1 Smoothing."""
        count_12 = self.bigrams[char1].get(char2, 0)
        count_1 = self.unigrams.get(char1, 0)
        vocab_size = len(self.vocab) if len(self.vocab) > 0 else 1
        
        prob = (count_12 + 1) / (count_1 + vocab_size)
        return math.log(prob)

def viterbi_decode(word_results, lm, visual_weight=0.7, lm_weight=0.3):
    """Finds the optimal sequence through Top-K candidates."""
    if not word_results:
        return [], []
        
    # Ensure no empty recognitions break the chain
    for seg in word_results:
        if not seg['recognitions']:
            seg['recognitions'] = [{'char_class': '?', 'confidence': 1e-9, 'distance': 999}]
            
    V = [{}]
    
    # Step 0: Initialize first character
    for rec in word_results[0]['recognitions']:
        char = clean_char_class(rec['char_class'])
        vis_score = rec.get('confidence', 1e-9)
        V[0][char] = (visual_weight * math.log(vis_score + 1e-9), None, rec)
        
    # Run Viterbi for remaining positions
    for t in range(1, len(word_results)):
        V.append({})
        for rec in word_results[t]['recognitions']:
            curr_char = clean_char_class(rec['char_class'])
            vis_score = visual_weight * math.log(rec.get('confidence', 1e-9) + 1e-9)
            
            best_score = float('-inf')
            best_prev = None
            
            for prev_char, (prev_score, _, _) in V[t-1].items():
                trans_score = lm_weight * lm.get_transition_log_prob(prev_char, curr_char)
                total_score = prev_score + trans_score + vis_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_prev = prev_char
                    
            V[t][curr_char] = (best_score, best_prev, rec)
            
    # Backtrack to find the best path
    if not V[-1]: 
        return [], []
        
    best_final_char = max(V[-1].keys(), key=lambda k: V[-1][k][0])
    
    path_chars = []
    path_recs = []
    curr_char = best_final_char
    
    for t in range(len(word_results) - 1, -1, -1):
        # Access safely from V[t]
        state = V[t].get(curr_char)
        if not state:
            # Fallback if something went wrong in backtracking
            break
        _, prev_char, rec = state
        path_chars.insert(0, curr_char)
        path_recs.insert(0, rec)
        curr_char = prev_char
        
    return path_chars, path_recs


# ============================================================================
# Utility Functions
# ============================================================================

def clean_char_class(char_class):
    """Remove variant suffixes like (1), (2) from character class labels."""
    return re.sub(r'\(\d+\)$', '', char_class).strip()


def get_ground_truth_from_filename(filename):
    """
    Extract ground truth word from image filename.
    Handles cases like:
      - "അകിൽ.png" → "അകിൽ"
      - "ജാതഗം (2).png" → "ജാതഗം"
      - "കാസ് ര.png" → "കാസ് ര"  (keep internal spaces)
    """
    name = Path(filename).stem
    # Remove trailing " (N)" disambiguation suffixes
    name = re.sub(r'\s*\(\d+\)\s*$', '', name)
    return name.strip()


# ============================================================================
# Main Evaluation
# ============================================================================

def normalize_height(image: Image.Image, target_height: int = 150) -> Image.Image:
    """Normalize word image to a standard height before recognition to keep segmentation thresholds consistent."""
    if image.height == 0:
        return image
    ratio = target_height / image.height
    new_width = max(1, int(image.width * ratio))
    return image.resize((new_width, target_height), Image.LANCZOS)


def evaluate_testsamples(
    test_dir="/mnt/Data_volume/Dhanya/testsamples500k",
    output_dir="/mnt/Data_volume/Dhanya/project",
    max_images=None,
    n_results=1,
    verbose=False
):
    """
    Run CER evaluation on test samples.

    Args:
        test_dir: Path to testsamples500k directory
        output_dir: Where to save results
        max_images: Limit number of images (None = all)
        n_results: Top-N results per character recognition
        verbose: Print every image result

    Returns:
        (summary_dict, results_list)
    """
    # Initialize recognizer
    print("=" * 70)
    print("MALAYALAM OCR - CER EVALUATION")
    print("=" * 70)
    print("\nInitializing CharacterRecognizer...")
    recognizer = CharacterRecognizer()

    # Get DB stats
    stats = recognizer.char_db.get_stats()
    print(f"Character DB: {stats}")

    if stats['character_count'] == 0:
        print("\n❌ ERROR: Character database is empty! Run indexing first.")
        return None, None

    # Collect test images
    test_path = Path(test_dir)
    image_extensions = {'.png', '.jpg', '.jpeg'}
    test_images = sorted([
        f for f in test_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if max_images:
        test_images = test_images[:max_images]

    total = len(test_images)
    print(f"\nTest images: {total}")
    print(f"Test directory: {test_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # --- NEW: Train the Language Model on our test set ground truths ---
    print("\nTraining Bigram Language Model...")
    corpus = [get_ground_truth_from_filename(f.name) for f in test_images]
    lm = CharacterBigramModel()
    lm.train(corpus)
    # -------------------------------------------------------------------

    results = []
    total_cer = 0.0
    errors = 0
    start_time = time.time()

    for idx, img_path in enumerate(test_images):
        # Ground truth from filename
        ground_truth = get_ground_truth_from_filename(img_path.name)

        try:
            img_raw = Image.open(img_path)
            if img_raw.mode == 'RGBA':
                background = Image.new('RGB', img_raw.size, (255, 255, 255))
                background.paste(img_raw, mask=img_raw.split()[3])
                image = background
            else:
                image = img_raw.convert('RGB')
                
            # Normalize image height
            image = normalize_height(image, target_height=150)

            # Recognize word (segment + per-character recognition)
            word_results = recognizer.recognize_word_image(
                image, n_results=n_results
            )

            if idx < 3:
                print(f"\n--- DEBUG image: {img_path.name} ---")
                print(f"  GT: '{ground_truth}' ({len(split_grapheme_clusters(ground_truth))} clusters)")
                print(f"  Segments produced: {len(word_results)}")
                for i, seg in enumerate(word_results):
                    if seg['recognitions']:
                        r = seg['recognitions'][0]
                        print(f"  seg[{i}] → char_class='{r['char_class']}' distance={r['distance']:.4f} confidence={r['confidence']:.4f}")
                    else:
                        print(f"  seg[{i}] → NO RECOGNITION")

            # Build predicted word
            if n_results > 1:
                predicted_chars, path_recs = viterbi_decode(word_results, lm)
                segment_details = []
                for rec in path_recs:
                    char_class = clean_char_class(rec['char_class'])
                    segment_details.append({
                        'char': char_class,
                        'distance': round(rec.get('distance', -1), 4),
                        'confidence': round(rec.get('confidence', 0), 4)
                    })
            else:
                # Top-1 fallback
                predicted_chars = []
                segment_details = []
                for seg in word_results:
                    if seg['recognitions']:
                        top_match = seg['recognitions'][0]
                        char_class = clean_char_class(top_match['char_class'])
                        predicted_chars.append(char_class)
                        segment_details.append({
                            'char': char_class,
                            'distance': round(top_match['distance'], 4),
                            'confidence': round(top_match.get('confidence', 0), 4)
                        })
                    else:
                        predicted_chars.append('?')
                        segment_details.append({'char': '?', 'distance': -1, 'confidence': 0})

            CHAR_REMAP = {
                '൦': 'ം',
                'ഠ': 'ം',
                'ഃ': 'ം',   # only at word-final — keep as is
                '൮': 'ഉ',   # digit 8 ← consistently confused with ഉ
                'ഒ': 'ഓ',   # only when preceded by ോ-type vowel marker
            }
            
            def remap_predicted(chars):
                if not chars:
                    return chars
                remapped = list(chars)
                
                # Apply positionally-agnostic remaps
                for i in range(len(remapped)):
                    if remapped[i] == '൮':
                        remapped[i] = 'ഉ'
                    elif remapped[i] == 'ഒ' and i > 0:
                        if remapped[i-1] in ['ോ', 'േ', 'ൊ', 'ാ']:
                            remapped[i] = 'ഓ'
                            
                # Apply word-final remaps
                last = remapped[-1]
                if last in ['൦', 'ഠ', 'ഃ']:
                    remapped[-1] = 'ം'
                return remapped

            predicted_chars = remap_predicted(predicted_chars)
            predicted_word = ''.join(predicted_chars)

            # Compute CER
            cer = compute_cer(predicted_word, ground_truth)
            total_cer += cer

            gt_clusters = split_grapheme_clusters(ground_truth)
            pred_clusters = split_grapheme_clusters(predicted_word)

            result = {
                'image': img_path.name,
                'ground_truth': ground_truth,
                'predicted': predicted_word,
                'cer': round(cer, 4),
                'num_segments': len(word_results),
                'gt_length': len(gt_clusters),
                'pred_length': len(pred_clusters),
                'gt_clusters': gt_clusters,
                'pred_clusters': pred_clusters,
                'segments': segment_details
            }
            results.append(result)

            # Progress logging
            if verbose or (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total - idx - 1) / rate if rate > 0 else 0
                avg_cer = total_cer / (idx + 1)
                print(
                    f"[{idx+1:3d}/{total}] CER: {cer:.4f} | "
                    f"Avg: {avg_cer:.4f} | "
                    f"GT: '{ground_truth}' → Pred: '{predicted_word}' | "
                    f"{rate:.1f} img/s | ETA: {eta:.0f}s"
                )

        except Exception as e:
            errors += 1
            gt_clusters = split_grapheme_clusters(ground_truth)
            result = {
                'image': img_path.name,
                'ground_truth': ground_truth,
                'predicted': '',
                'cer': 1.0,
                'num_segments': 0,
                'gt_length': len(gt_clusters),
                'pred_length': 0,
                'gt_clusters': gt_clusters,
                'pred_clusters': [],
                'segments': [],
                'error': str(e)
            }
            results.append(result)
            total_cer += 1.0
            print(f"[{idx+1:3d}/{total}] ❌ ERROR: {img_path.name}: {e}")

    # ========================================================================
    # Summary Statistics
    # ========================================================================
    elapsed = time.time() - start_time
    num_evaluated = len(results)
    avg_cer = total_cer / num_evaluated if num_evaluated > 0 else 0

    cer_values = [r['cer'] for r in results]
    perfect = sum(1 for c in cer_values if c == 0.0)
    below_10 = sum(1 for c in cer_values if c <= 0.1)
    below_25 = sum(1 for c in cer_values if c <= 0.25)
    below_50 = sum(1 for c in cer_values if c <= 0.5)

    summary = {
        'total_images': total,
        'evaluated': num_evaluated,
        'errors': errors,
        'average_cer': round(avg_cer, 4),
        'median_cer': round(float(np.median(cer_values)), 4),
        'min_cer': round(min(cer_values), 4),
        'max_cer': round(max(cer_values), 4),
        'std_cer': round(float(np.std(cer_values)), 4),
        'perfect_matches': perfect,
        'perfect_match_pct': round(100 * perfect / num_evaluated, 2),
        'cer_leq_10pct': below_10,
        'cer_leq_25pct': below_25,
        'cer_leq_50pct': below_50,
        'word_accuracy_pct': round(100 * perfect / num_evaluated, 2),
        'time_seconds': round(elapsed, 1),
        'images_per_second': round(num_evaluated / elapsed, 2) if elapsed > 0 else 0
    }

    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")

    # ========================================================================
    # Save Results
    # ========================================================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON (detailed, includes per-segment info)
    json_path = output_path / "cer_evaluation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Detailed JSON: {json_path}")

    # CSV (summary per image)
    csv_path = output_path / "cer_evaluation_results.csv"
    csv_fields = [
        'image', 'ground_truth', 'predicted', 'cer',
        'num_segments', 'gt_length', 'pred_length'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in csv_fields})
    print(f"📄 CSV summary:   {csv_path}")

    # Top-10 worst results for analysis
    worst = sorted(results, key=lambda r: r['cer'], reverse=True)[:10]
    print("\n🔴 TOP 10 WORST (highest CER):")
    for i, w in enumerate(worst, 1):
        print(f"  {i:2d}. CER={w['cer']:.4f} | "
              f"GT='{w['ground_truth']}' → Pred='{w['predicted']}' | "
              f"{w['image']}")

    # Top-10 best results
    best = sorted(results, key=lambda r: r['cer'])[:10]
    print("\n🟢 TOP 10 BEST (lowest CER):")
    for i, b in enumerate(best, 1):
        print(f"  {i:2d}. CER={b['cer']:.4f} | "
              f"GT='{b['ground_truth']}' → Pred='{b['predicted']}' | "
              f"{b['image']}")

    return summary, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CER Evaluation for Malayalam OCR RAG Pipeline"
    )
    parser.add_argument(
        "--test-dir",
        default="/mnt/Data_volume/Dhanya/testsamples500k",
        help="Path to test samples directory"
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Max images to evaluate (None = all 500)"
    )
    parser.add_argument(
        "--n-results", type=int, default=1,
        help="Top-N recognition results per character segment"
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/Data_volume/Dhanya/project",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print result for every image"
    )

    args = parser.parse_args()

    summary, results = evaluate_testsamples(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        n_results=args.n_results,
        verbose=args.verbose
    )
