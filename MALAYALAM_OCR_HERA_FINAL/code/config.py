"""
Configuration for Multimodal RAG System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ma"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VAL_DIR = DATA_DIR / "val"
VOCAB_FILE = DATA_DIR / "vocab.txt"
TRAIN_MAPPING = DATA_DIR / "train.txt"
TEST_MAPPING = DATA_DIR / "test.txt"
VAL_MAPPING = DATA_DIR / "val.txt"

# ChromaDB
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "malayalam_handwriting"

# Character Datasets (Local folders - deprecated)
COMPOUND_MAL_DIR = BASE_DIR / "Compound_MAL_characters" / "EXPANNEW"
PARTS_KAYYEZHUTHU_DIR = BASE_DIR / "P-ARTS KAYYEZHUTHU MALAYALAM"
MALAYALAM_RAG_DATASET_DIR = BASE_DIR / "malayalam_rag_dataset"
CHAR_COLLECTION_NAME = "malayalam_characters"

# Character Dataset (Primary - 330k balanced samples, 660 character labels)
LOCAL_CHAR_DATASET = "/mnt/Data_volume/Dhanya/project/malayalam_character_balanced"
HF_CHAR_DATASET = LOCAL_CHAR_DATASET  # Legacy alias, now points to local path
USE_HF_DATASET = True  # Set to True to use dataset-based indexing instead of local folders

# HuggingFace Token
HF_TOKEN = os.getenv("HF_TOKEN")

# CLIP/SigLIP Model - Using fine-tuned naflex model
CLIP_MODEL_NAME = "/mnt/Data_volume/Dhanya/project/siglip_malayalam_finetuned/merged_model"
# Gemini
GEMINI_MODEL = "gemini-2.0-flash"

# Google Cloud / Vertex AI Config
GCP_PROJECT_ID = "noted-computing-459609-n2"
GCP_LOCATION = "us-central1"

# Select Backend: "clip" or "gemini"
EMBEDDING_BACKEND = "clip" 

# Hybrid Embeddings: SigLIP + HOG for better character recognition
USE_HYBRID_EMBEDDINGS = True
HOG_WEIGHT = 0.2  # 20% HOG, 80% SigLIP
SIGLIP_DIM = 768
HOG_DIM = 1764  # HOG with (9 orientations, 8x8 pixels per cell, 2x2 cells per block) on 64x64 image

if EMBEDDING_BACKEND == "gemini":
    EMBEDDING_DIM = 1408
    CHAR_COLLECTION_NAME = "malayalam_characters_gemini"
    BATCH_SIZE = 1 # Force low batch size for API calls
elif USE_HYBRID_EMBEDDINGS:
    EMBEDDING_DIM = SIGLIP_DIM + HOG_DIM  # 768 + 1764 = 2532
    CHAR_COLLECTION_NAME = "malayalam_characters_hybrid"  # New collection for hybrid embeddings
    BATCH_SIZE = 32  # Smaller batch for hybrid
else:
    EMBEDDING_DIM = SIGLIP_DIM
    CHAR_COLLECTION_NAME = "malayalam_characters_naflex"
    BATCH_SIZE = 64

# Preprocessing target size (matches CLIP's native resolution)
PREPROCESSING_TARGET_SIZE = 224

# HOG Settings
HOG_IMAGE_SIZE = 64  # Resize to 64x64 for HOG
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# DINOv2 Model for Reranking (disabled when using hybrid)
DINO_MODEL_NAME = "facebook/dinov2-large"
RERANKING_MODEL = "none"  # Disabled - hybrid embeddings provide better results

