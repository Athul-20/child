"""
Character-Level Malayalam OCR using CLIP Embeddings
Stores individual character images as embeddings for recognition via similarity search.

IMPORTANT: Images are preprocessed consistently for both indexing and recognition
to ensure embeddings are comparable. See preprocessing.py for details.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import chromadb
import gc
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import re
import torch
from collections import defaultdict
import config
from embeddings import CLIPEmbedder
from preprocessing import preprocess_character_image, preprocess_images_batch, binarize_image
try:
    from gemini_embeddings import GeminiEmbedder
except ImportError:
    GeminiEmbedder = None # Type placeholder
from segmentation import segment_word
from embeddings import extract_hog_features


class CharacterVectorDB:
    """ChromaDB-based vector store specifically for character embeddings"""
    
    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or str(config.CHROMA_PERSIST_DIR)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Get or create character collection (separate from word collection)
        self.collection = self.client.get_or_create_collection(
            name=config.CHAR_COLLECTION_NAME,
            metadata={"description": "Malayalam character embeddings for OCR"}
        )
        
        print(f"Character Collection '{config.CHAR_COLLECTION_NAME}' initialized")
        print(f"Current character count: {self.collection.count()}")
    
    def add_characters(self, 
                       ids: List[str],
                       embeddings: List[List[float]], 
                       metadatas: List[Dict],
                       batch_size: int = 100):
        """Add character embeddings to the collection."""
        total = len(ids)
        
        for i in tqdm(range(0, total, batch_size), desc="Adding to CharacterDB"):
            end_idx = min(i + batch_size, total)
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        print(f"Added {total} characters. Total count: {self.collection.count()}")
    
    def search(self, 
               query_embedding: List[float], 
               n_results: int = 5) -> Dict:
        """Search for similar characters."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            "collection_name": config.CHAR_COLLECTION_NAME,
            "character_count": self.collection.count(),
            "persist_dir": self.persist_dir
        }
    
    def count(self) -> int:
        """Return total number of characters in the database"""
        return self.collection.count()

    def clear(self):
        """Clear all characters from collection"""
        self.client.delete_collection(config.CHAR_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.CHAR_COLLECTION_NAME,
            metadata={"description": "Malayalam character embeddings for OCR"}
        )
        print("Character collection cleared")


# ============================================================================
# Language Modeling & Viterbi Decoding (HERA Optimized Pipeline)
# ============================================================================

import math

def split_grapheme_clusters(text):
    """
    Split Malayalam text into visual character units (grapheme clusters).
    Handles base characters followed by virama, vowel signs, and anusvara.
    """
    if not text:
        return []
    # Simplified regex for visual units: 
    # Base + (Virama + Base)* + (Vowel Signs | Anusvara | Visarga)?
    # This captures the visual intuition of a "character" in OCR.
    pattern = r'[\u0D00-\u0D7F][\u0D4D\u0D3E-\u0D4C\u0D02\u0D03]*'
    return re.findall(pattern, text)


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
        return self

    def get_transition_log_prob(self, char1, char2):
        """Returns log probability with Add-1 Smoothing."""
        count_12 = self.bigrams[char1].get(char2, 0)
        count_1 = self.unigrams.get(char1, 0)
        vocab_size = len(self.vocab) if len(self.vocab) > 0 else 1
        prob = (count_12 + 0.1) / (count_1 + vocab_size * 0.1)
        return math.log(prob)


def load_malayalam_lm():
    """Initializes and trains the language model on the Gold Standard corpus."""
    gold_dir = "/mnt/Data_volume/Dhanya/project/GRAND_GOLD_STANDARD_200"
    corpus = []
    if os.path.exists(gold_dir):
        from pathlib import Path
        for f in os.listdir(gold_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                name = Path(f).stem
                name = re.sub(r'\s*\(\d+\)\s*$', '', name).strip()
                corpus.append(name)
    
    lm = CharacterBigramModel()
    if corpus:
        lm.train(corpus)
    return lm


def clean_char_class(char_class):
    """Remove variant suffixes like (1), (2) from character class labels."""
    return re.sub(r'^(HF_|PARTS_)|\(\d+\)$', '', char_class).strip()


def viterbi_decode(word_results, lm, visual_weight=0.7, lm_weight=0.3):
    """Finds the optimal sequence through Top-K candidates using Viterbi algorithm."""
    if not word_results:
        return [], []
        
    for seg in word_results:
        if not seg['recognitions']:
            seg['recognitions'] = [{'char_class': '?', 'confidence': 1e-9, 'distance': 999}]
            
    V = [{}]
    
    # Step 0: Initialize
    for rec in word_results[0]['recognitions']:
        char = clean_char_class(rec['char_class'])
        vis_score = rec.get('confidence', 1e-9)
        V[0][char] = (visual_weight * math.log(vis_score + 1e-9), None, rec)
        
    # Run Viterbi
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
            
    if not V[-1]: return [], []
        
    best_final_char = max(V[-1].keys(), key=lambda k: V[-1][k][0])
    path_chars, path_recs = [], []
    curr_char = best_final_char
    
    for t in range(len(word_results) - 1, -1, -1):
        state = V[t].get(curr_char)
        if not state: break
        _, prev_char, rec = state
        path_chars.insert(0, curr_char)
        path_recs.insert(0, rec)
        curr_char = prev_char
        
    return path_chars, path_recs


def _indexing_worker(item):
    """Parallel worker for indexing: Preprocess + HOG"""
    try:
        # Preprocess
        from preprocessing import format_for_siglip
        preprocessed = format_for_siglip(item["image"])
        
        # HOG (Structural features)
        hog_vec = None
        if getattr(config, 'USE_HYBRID_EMBEDDINGS', False):
            hog_vec = extract_hog_features(preprocessed)
            
        return {
            "image": preprocessed,
            "hog": hog_vec,
            "label": item["label"],
            "idx": item["idx"]
        }
    except Exception:
        from PIL import Image
        # Fallback for corrupted images
        blank = Image.new('RGB', (config.PREPROCESSING_TARGET_SIZE, config.PREPROCESSING_TARGET_SIZE), (255, 255, 255))
        return {
            "image": blank,
            "hog": np.zeros(getattr(config, 'HOG_DIM', 1764)),
            "label": item.get("label", "unknown"),
            "idx": item.get("idx", 0)
        }


def _indexing_worker_thread(item):
    """Thread-safe worker for indexing: Preprocess + HOG.
    Uses ThreadPoolExecutor instead of ProcessPoolExecutor to avoid
    expensive PIL image pickling across process boundaries.
    """
    try:
        from preprocessing import format_for_siglip
        preprocessed = format_for_siglip(item["image"])
        
        hog_vec = None
        if getattr(config, 'USE_HYBRID_EMBEDDINGS', False):
            hog_vec = extract_hog_features(preprocessed)
            
        return {
            "image": preprocessed,
            "hog": hog_vec,
            "label": item["label"],
            "idx": item["idx"]
        }
    except Exception:
        from PIL import Image
        blank = Image.new('RGB', (config.PREPROCESSING_TARGET_SIZE, config.PREPROCESSING_TARGET_SIZE), (255, 255, 255))
        return {
            "image": blank,
            "hog": np.zeros(getattr(config, 'HOG_DIM', 1764)),
            "label": item.get("label", "unknown"),
            "idx": item.get("idx", 0)
        }


def scan_character_folders(base_dir: Path) -> List[Tuple[str, str, List[str]]]:
    """
    Scan a character dataset directory and return folder info.
    
    Returns:
        List of (char_class, folder_path, image_paths) tuples
    """
    results = []
    
    if not base_dir.exists():
        print(f"Warning: Directory not found: {base_dir}")
        return results
    
    # Find all CHAR* folders
    char_folders = sorted([f for f in base_dir.iterdir() if f.is_dir() and f.name.startswith("CHAR")])
    
    for folder in char_folders:
        char_class = folder.name
        # Get all image files in the folder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend([str(p) for p in folder.rglob(ext)])
        
        if image_files:
            results.append((char_class, str(folder), image_files))
    
    return results


def index_character_dataset(
    limit_per_class: Optional[int] = None,
    clear_existing: bool = False
):
    """
    Index all character images into the vector database.
    
    If config.USE_HF_DATASET is True, uses HuggingFace dataset.
    Otherwise, uses local folder-based datasets.
    
    Args:
        limit_per_class: Max images per character class (for testing). None = all images.
        clear_existing: Whether to clear existing data before indexing.
    """
    # Check if we should use HuggingFace dataset
    if getattr(config, 'USE_HF_DATASET', False):
        return index_from_huggingface(
            limit_per_class=limit_per_class,
            clear_existing=clear_existing
        )
    
    # Otherwise, use local folder-based indexing (legacy)
    print("\n" + "="*60)
    print("INDEXING MALAYALAM CHARACTER DATASETS (Local Folders)")
    print("="*60)
    
    # Initialize embedder and vector DB
    if config.EMBEDDING_BACKEND == "gemini":
        print("Using backend: Gemini (Vertex AI)")
        embedder = GeminiEmbedder()
    else:
        print("Using backend: CLIP")
        embedder = CLIPEmbedder()
        
    char_db = CharacterVectorDB()
    
    if clear_existing:
        char_db.clear()
    
    # Collect all character folders from both datasets
    all_char_data = []
    
    # Dataset 1: Compound MAL Characters
    print(f"\nScanning: {config.COMPOUND_MAL_DIR}")
    compound_chars = scan_character_folders(config.COMPOUND_MAL_DIR)
    for char_class, folder_path, images in compound_chars:
        for img_path in images[:limit_per_class] if limit_per_class else images:
            all_char_data.append({
                "char_class": char_class,
                "image_path": img_path,
                "dataset": "compound_mal"
            })
    print(f"  Found {len(compound_chars)} character classes")
    
    # Dataset 2: P-ARTS Kayyezhuthu
    print(f"\nScanning: {config.PARTS_KAYYEZHUTHU_DIR}")
    parts_chars = scan_character_folders(config.PARTS_KAYYEZHUTHU_DIR)
    for char_class, folder_path, images in parts_chars:
        for img_path in images[:limit_per_class] if limit_per_class else images:
            all_char_data.append({
                "char_class": f"PARTS_{char_class}",  # Prefix to distinguish
                "image_path": img_path,
                "dataset": "parts_kayyezhuthu"
            })
    print(f"  Found {len(parts_chars)} character classes")
    
    total_images = len(all_char_data)
    print(f"\nTotal images to index: {total_images}")
    
    if total_images == 0:
        print("No images found! Check dataset paths.")
        return
    
    # Extract image paths for embedding
    image_paths = [item["image_path"] for item in all_char_data]
    
    # Generate embeddings with inline preprocessing
    print(f"\nGenerating {config.EMBEDDING_BACKEND.upper()} embeddings with preprocessing...")
    print("(Each image: load → grayscale → binarize → center → resize 224x224 → embed)")
    embeddings = embedder.embed_images_batch_with_preprocessing(image_paths, batch_size=config.BATCH_SIZE)
    
    # Prepare data for insertion
    ids = [f"char_{i}" for i in range(len(all_char_data))]
    metadatas = [
        {
            "char_class": item["char_class"],
            "image_path": item["image_path"],
            "dataset": item["dataset"]
        }
        for item in all_char_data
    ]
    
    # Add to vector DB
    print("\nAdding to character vector database...")
    char_db.add_characters(ids, embeddings, metadatas)
    
    print("\n" + "="*60)
    print("INDEXING COMPLETE!")
    print("="*60)
    print(char_db.get_stats())


def index_from_huggingface(
    limit_per_class: Optional[int] = None,
    clear_existing: bool = False,
    max_samples: Optional[int] = None,
    chunk_size: int = 25000,
    gpu_batch_size: int = 256,
    cpu_workers: int = 16
):
    """
    HIGH-PERFORMANCE indexing from HuggingFace dataset.
    Optimized for H100 GPU and 16-core CPU.
    
    Optimizations:
    - ThreadPoolExecutor for CPU preprocessing (avoids PIL pickling overhead)
    - dataset.select() for batch loading (avoids one-by-one access)
    - Persistent DB connection across chunks
    - Larger chunk and batch sizes
    """
    from datasets import load_from_disk
    
    # 1. Check current count first using a temporary DB connection
    print("\n" + "="*60)
    print("🚀 HIGH-PERFORMANCE H100 INDEXING")
    print(f"Dataset: {config.LOCAL_CHAR_DATASET}")
    print(f"Chunk size: {chunk_size:,} | GPU batch: {gpu_batch_size}")
    print("="*60)
    
    char_db = CharacterVectorDB()
    if clear_existing:
        print("Clearing existing data...")
        char_db.clear()
        
    current_count = char_db.count()
    print(f"Current DB Count: {current_count:,}")
    
    # Keep DB connection open (no more del/re-open per chunk)
    
    embedder = None # Defer initialization to avoid Segfault with multi-processing
    
    # 2. Load Dataset (Disable internal parallelism to avoid conflicts)
    print(f"\nLoading dataset from disk: {config.LOCAL_CHAR_DATASET}")
    dataset = load_from_disk(config.LOCAL_CHAR_DATASET)
    total_rows = len(dataset)
    print(f"Dataset loaded: {total_rows:,} samples")
    
    # Load Character Mapping if exists
    char_names = {}
    mapping_path = Path(config.BASE_DIR) / "char_names.txt"
    if mapping_path.exists():
        import ast
        with open(mapping_path, "r", encoding="utf-8") as f:
            char_names = ast.literal_eval(f.read())
        print(f"Loaded {len(char_names)} character mappings.")

    # 3. Apply Limits & Resume
    start_idx = current_count
    end_idx = min(max_samples, total_rows) if max_samples else total_rows
    
    if start_idx >= end_idx:
        print(f"Nothing to index. (DB count {current_count} >= Goal {end_idx})")
        return None

    print(f"Resuming from index: {start_idx:,} to {end_idx:,}")
    total_to_process = end_idx - start_idx
    overall_start = time.time()
    
    # 4. Main Chunked Pipeline
    for chunk_start in range(start_idx, end_idx, chunk_size):
        this_chunk_end = min(chunk_start + chunk_size, end_idx)
        chunk_len = this_chunk_end - chunk_start
        print(f"\n📦 Processing Chunk: {chunk_start:,} to {this_chunk_end:,} ({chunk_len:,} samples)")
        
        # FAST: Use dataset.select() for batch loading instead of one-by-one access
        chunk_indices = list(range(chunk_start, this_chunk_end))
        chunk_dataset = dataset.select(chunk_indices)
        
        # Prepare items for workers
        chunk_items = []
        for i, item in enumerate(chunk_dataset):
            item["idx"] = chunk_start + i
            chunk_items.append(item)
            
        # STEP A: Parallel CPU work (Preprocess + HOG) using ThreadPoolExecutor
        # ThreadPoolExecutor avoids the expensive pickling of PIL images that
        # ProcessPoolExecutor requires. OpenCV/numpy release the GIL anyway.
        print(f"  Step 1: CPU Pipeline ({cpu_workers} threads)...")
        start_t = time.time()
        with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
            results = list(tqdm(executor.map(_indexing_worker_thread, chunk_items), 
                                total=len(chunk_items), desc="  Processing", leave=False))
        
        cpu_time = time.time() - start_t
        print(f"  Done in {cpu_time:.2f}s ({len(results)/cpu_time:.1f} samples/sec)")
        
        # Free chunk data early
        del chunk_items, chunk_dataset
        
        # STEP B: GPU Embedding (with CPU<->GPU pipelining inside the embedder)
        print(f"  Step 2: GPU Embedding (H100, FP16, pipelined)...")
        if embedder is None:
            if config.EMBEDDING_BACKEND == "gemini":
                embedder = GeminiEmbedder()
            else:
                embedder = CLIPEmbedder()
        
        images = [r["image"] for r in results]
        hogs = [r["hog"] for r in results]
        labels = [r["label"] for r in results]
        
        start_t = time.time()
        # Use the optimized pipelined function
        if hasattr(embedder, 'embed_images_batch_hybrid_precalculated') and (hogs[0] is not None):
            embeddings = embedder.embed_images_batch_hybrid_precalculated(
                images, hogs, batch_size=gpu_batch_size
            )
        else:
            embeddings = embedder.embed_images_batch(images, batch_size=gpu_batch_size)
            
        embed_time = time.time() - start_t
        print(f"  Done in {embed_time:.2f}s ({len(images)/embed_time:.1f} samples/sec)")
        
        # Free images early (large memory)
        del images, hogs
        
        # STEP C: DB Insertion
        print(f"  Step 3: Storing in DB...")
        ids = [f"balanced_{r['idx']}" for r in results]
        metadatas = []
        for label in labels:
            if isinstance(label, int):
                if isinstance(char_names, dict):
                    char_class = char_names.get(label, f"class_{label}")
                elif isinstance(char_names, list) and label < len(char_names):
                    char_class = char_names[label]
                else:
                    char_class = f"class_{label}"
            else:
                char_class = label
            metadatas.append({
                "char_class": char_class,
                "dataset": "balanced",
                "source_dataset": config.HF_CHAR_DATASET
            })
        char_db.add_characters(ids, embeddings, metadatas, batch_size=5000)
        
        # Memory Cleanup (keep DB open)
        del results, embeddings, labels, metadatas, ids
        gc.collect()
        torch.cuda.empty_cache()
        
        current_count = char_db.count()
        elapsed = time.time() - overall_start
        processed = current_count - start_idx
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total_to_process - processed) / rate if rate > 0 else 0
        print(f"  ✅ DB Count: {current_count:,} | Rate: {rate:.1f} samples/sec | ETA: {remaining/60:.1f} min")

    total_time = time.time() - overall_start
    print("\n" + "="*60)
    print("✅ INDEXING COMPLETE!")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"Average rate: {total_to_process/total_time:.1f} samples/sec")
    print("="*60)
    final_stats = char_db.get_stats()
    print(final_stats)
    return final_stats


class CharacterRecognizer:
    """Recognize Malayalam characters using embedding similarity."""
    
    def __init__(self, model_name=None):
        if config.EMBEDDING_BACKEND == "gemini":
            self.embedder = GeminiEmbedder()
        else:
            self.embedder = CLIPEmbedder(model_name=model_name)
            
        self.char_db = CharacterVectorDB()
        
    def preprocess_query_image(self, image):
        """
        Preprocess query image to match indexed database images.
        Uses the shared format function to avoid destroying connected components.
        """
        from preprocessing import format_for_siglip
        return format_for_siglip(image)
        
    def recognize(self, image, n_results: int = 5, distance_threshold: float = 1.5, use_reranking: bool = False) -> List[Dict]:
        """
        Recognize a character from an image using vector similarity + majority voting.
        
        Fetches many candidates, groups by char_class, and returns top classes
        based on weighted voting (weighted by inverse distance).
        
        Args:
            image: PIL Image or path to image file
            n_results: Number of top matches to return
            distance_threshold: Maximum distance to consider valid (lower = stricter)
            use_reranking: If True, apply majority voting reranking
            
        Returns:
            List of dicts with char_class, distance, confidence, image_path
        """
        from PIL import Image as PILImage
        from collections import defaultdict
        
        # Load image if path
        from preprocessing import to_rgb
        if isinstance(image, str):
            pil_image = to_rgb(PILImage.open(image))
        elif hasattr(image, 'convert'):
            pil_image = to_rgb(image)
        else:
            pil_image = image
        
        # Preprocess query image
        preprocessed = self.preprocess_query_image(pil_image)
        
        # Generate embedding for query image using the EXACT same pipeline as indexing
        # to ensure perfect mathematical alignment in the vector space
        from embeddings import extract_hog_features
        hog_feat = extract_hog_features(preprocessed)
        query_embedding = self.embedder.embed_images_batch_hybrid_precalculated(
            [preprocessed], 
            [hog_feat],
            batch_size=1
        )[0]        
        # Fetch many candidates for majority voting (50 candidates)
        initial_k = 50 if use_reranking else n_results
        
        # Search in character DB
        results = self.char_db.search(query_embedding, n_results=initial_k)
        
        # Process results
        candidates = []
        if results['ids'] and results['ids'][0]:
            for i, (id_, distance, metadata) in enumerate(zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0]
            )):
                # Convert distance to confidence (approximate)
                # Lower distance = higher confidence
                confidence = max(0.0, 1.0 - (distance / 2.0))
                
                candidates.append({
                    "rank": i + 1,
                    "char_class": metadata['char_class'],
                    "distance": distance,
                    "confidence": confidence,
                    "image_path": metadata.get('image_path'),
                    "dataset": metadata.get('dataset', 'unknown'),
                    "embedding_score": confidence,
                    "original_folder": metadata.get('original_folder'),
                })
        
        # Majority Voting with Weighted Scoring
        if use_reranking and candidates:
            # Group candidates by char_class and compute weighted score
            class_scores = defaultdict(lambda: {"total_weight": 0.0, "count": 0, "best_distance": float('inf'), "best_candidate": None})
            
            for cand in candidates:
                char_class = cand['char_class']
                distance = cand['distance']
                
                # Weight: inverse of distance (closer = higher weight)
                # Add small epsilon to avoid division by zero
                weight = 1.0 / (distance + 0.01)
                
                class_scores[char_class]["total_weight"] += weight
                class_scores[char_class]["count"] += 1
                
                # Track best (closest) candidate for this class
                if distance < class_scores[char_class]["best_distance"]:
                    class_scores[char_class]["best_distance"] = distance
                    class_scores[char_class]["best_candidate"] = cand
            
            # Sort classes by total weighted score
            sorted_classes = sorted(
                class_scores.items(),
                key=lambda x: x[1]["total_weight"],
                reverse=True
            )
            
            # Build result list with best candidate per class
            reranked = []
            for i, (char_class, scores) in enumerate(sorted_classes[:n_results]):
                best = scores["best_candidate"]
                if best:
                    best = best.copy()
                    best["rank"] = i + 1
                    # Normalize total weight to 0-1 range
                    max_possible = scores["count"] / 0.01  # Maximum if all had distance=0
                    best["confidence"] = min(1.0, scores["total_weight"] / (initial_k * 10))
                    best["vote_count"] = scores["count"]
                    best["vote_weight"] = round(scores["total_weight"], 2)
                    reranked.append(best)
            
            return reranked
        
        # Just return the top matches, but keep only the best per class
        unique_candidates = []
        seen_classes = set()
        for cand in candidates:
            if cand['char_class'] not in seen_classes:
                unique_candidates.append(cand)
                seen_classes.add(cand['char_class'])
            if len(unique_candidates) >= n_results:
                break
        return unique_candidates
    
    def recognize_word_image(self, image, n_results: int = 1, remove_lines: bool = False) -> List[Dict]:
        from segmentation import segment_word
        import numpy as np
        from PIL import Image
        
        def _normalize_height(img, target=150):
            if img.height == 0: return img
            ratio = target / img.height
            new_width = max(1, int(img.width * ratio))
            
            estimated_chars = max(1, new_width // 80)
            if new_width / estimated_chars < 60:
                target = 200  
                ratio = target / img.height
                new_width = max(1, int(img.width * ratio))
            return img.resize((new_width, target), Image.LANCZOS)
            
        # --- NEW: RELATIVE SPATIAL PASTING ---
        def _combine_images_spatially(seg1, seg2):
            """
            Combines two clean mask-extracted images onto a single white canvas 
            preserving their exact original spatial relativity.
            """
            b1 = seg1['bbox']
            b2 = seg2['bbox']
            
            # 1. Determine the size of the new combined bounding box
            min_x = min(b1[0], b2[0])
            min_y = min(b1[1], b2[1])
            max_x = max(b1[2], b2[2])
            max_y = max(b1[3], b2[3])
            
            new_w = max_x - min_x
            new_h = max_y - min_y
            
            # 2. Create a clean white canvas
            dst = Image.new('RGB', (new_w, new_h), (255, 255, 255))
            
            # 3. Paste the clean components using their relative offsets from min_x/min_y
            dst.paste(seg1['image'], (b1[0] - min_x, b1[1] - min_y))
            dst.paste(seg2['image'], (b2[0] - min_x, b2[1] - min_y))
            
            # Note: self.recognize() will automatically call format_for_siglip() 
            # to center/pad this combined image to 224x224 before embedding!
            return dst
        # --------------------    def recognize_word_image(self, image: Image.Image, n_results: int = 1, remove_lines=False):
        """
        Confidence-Driven Ensemble:
        Runs multiple segmentation strategies and picks the one with the highest 
        recognition confidence (lowest average distance).
        """
        from segmentation import (
            segment_word_cc_grouping, 
            segment_word_canny_contour, 
            segment_with_projection,
            enhance_user_image
        )
        
        # 1. Pre-enhance image
        image = _normalize_height(image)
        enhanced = enhance_user_image(image)
        
        # Define the processing logic for a set of segments
        def process_candidate_segments(char_images):
            # Convert images to dicts with bboxes if they aren't already
            char_data = []
            for img in char_images:
                char_data.append({"image": img, "bbox": (0, 0, img.width, img.height)})
            
            # --- INTERVENTION 1: Width-Ratio Pre-Merge ---
            if len(char_data) > 1:
                median_w = np.median([seg['bbox'][2] - seg['bbox'][0] for seg in char_data])
                merged_data = []
                i = 0
                while i < len(char_data):
                    seg = char_data[i]
                    w = seg['bbox'][2] - seg['bbox'][0]
                    if w < (median_w * 0.35) and i > 0:
                        prev = merged_data[-1]
                        combined_bbox = (
                            min(prev['bbox'][0], seg['bbox'][0]),
                            min(prev['bbox'][1], seg['bbox'][1]),
                            max(prev['bbox'][2], seg['bbox'][2]),
                            max(prev['bbox'][3], seg['bbox'][3])
                        )
                        merged_data[-1] = {
                            'image': _combine_images_spatially(prev, seg), 
                            'bbox': combined_bbox
                        }
                    else:
                        merged_data.append(seg)
                    i += 1
                char_data = merged_data

            # --- INTERVENTION 2: Conditional Semantic Merge ---
            final_segments = []
            i = 0
            MERGE_TRIGGER_DIST = 0.6
            MERGE_ACCEPT_DROP = 0.15

            while i < len(char_data):
                curr_seg = char_data[i]
                curr_recs = self.recognize(curr_seg["image"], n_results=n_results, use_reranking=False)
                curr_dist = curr_recs[0]['distance'] if curr_recs else float('inf')
                
                if curr_dist > MERGE_TRIGGER_DIST and i < len(char_data) - 1:
                    next_seg = char_data[i+1]
                    next_recs = self.recognize(next_seg["image"], n_results=1, use_reranking=False)
                    next_dist = next_recs[0]['distance'] if next_recs else float('inf')
                    
                    merged_img = _combine_images_spatially(curr_seg, next_seg)
                    merged_recs = self.recognize(merged_img, n_results=n_results, use_reranking=False)
                    merged_dist = merged_recs[0]['distance'] if merged_recs else float('inf')
                    
                    worse_of_two = max(curr_dist, next_dist)
                    if merged_dist < (worse_of_two - MERGE_ACCEPT_DROP):
                        final_segments.append({
                            "segment_index": len(final_segments),
                            "segment_image": merged_img,
                            "recognitions": merged_recs
                        })
                        i += 2
                        continue
                
                final_segments.append({
                    "segment_index": len(final_segments),
                    "segment_image": curr_seg["image"],
                    "recognitions": curr_recs
                })
                i += 1
            return final_segments

        # ----------------------------------------------------------------
        # ENSEMBLE EXECUTION
        # ----------------------------------------------------------------
        strategies = [
            ("CC Grouping", segment_word_cc_grouping),
            ("Canny Contour", segment_word_canny_contour),
            ("Projection", segment_with_projection)
        ]
        
        ensemble_candidates = []
        
        for name, strategy_func in strategies:
            try:
                chunks = strategy_func(enhanced)
                if not chunks:
                    continue
                    
                processed_results = process_candidate_segments(chunks)
                if not processed_results:
                    continue
                
                # Calculate average distance across all segments
                avg_dist = np.mean([r['recognitions'][0]['distance'] for r in processed_results if r['recognitions']])
                
                ensemble_candidates.append({
                    "name": name,
                    "results": processed_results,
                    "avg_distance": avg_dist
                })
            except Exception as e:
                print(f"Strategy {name} failed: {e}")
        
        if not ensemble_candidates:
            return []
            
        # Select candidate with lowest average distance
        best_candidate = min(ensemble_candidates, key=lambda x: x['avg_distance'])
        # print(f"  Ensemble selected: {best_candidate['name']} (Avg Dist: {best_candidate['avg_distance']:.4f})")
        
        return best_candidate['results']


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Index character datasets")
    parser.add_argument("--limit", type=int, default=None, help="Limit images per class")
    parser.add_argument("--clear", action="store_true", help="Clear existing data")
    parser.add_argument("--test", type=str, help="Test recognition with an image path")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    
    args = parser.parse_args()
    
    if args.index:
        index_character_dataset(
            limit_per_class=args.limit,
            clear_existing=args.clear
        )
    elif args.test:
        from PIL import Image
        recognizer = CharacterRecognizer()
        img = Image.open(args.test).convert('RGB')
        results = recognizer.recognize_word_image(img)
        print("\nRecognition Results:")
        pred = ""
        for seg in results:
            if seg['recognitions']:
                char = re.sub(r'^(HF_|PARTS_)|\(\d+\)$', '', seg['recognitions'][0]['char_class'])
                pred += char
        print(f"Predicted Word: {pred}")
        for i, seg in enumerate(results):
            if seg['recognitions']:
                r = seg['recognitions'][0]
                char = re.sub(r'^(HF_|PARTS_)|\(\d+\)$', '', r['char_class'])
                print(f"  Seg {i}: {char} (dist: {r['distance']:.4f})")
    elif args.stats:
        db = CharacterVectorDB()
        print(db.get_stats())
    else:
        parser.print_help()
