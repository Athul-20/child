"""
CLIP/SigLIP Embeddings with optional HOG Hybrid for Multimodal RAG
Supports hybrid embeddings: SigLIP (semantic) + HOG (structural)
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from skimage.feature import hog
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import config


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def extract_hog_features(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """
    Extract HOG features using fast OpenCV resizing.
    """
    # Convert to grayscale numpy array
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to HOG_IMAGE_SIZE x HOG_IMAGE_SIZE (using OpenCV)
    image = cv2.resize(image, (config.HOG_IMAGE_SIZE, config.HOG_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    
    # Extract HOG features
    hog_features = hog(
        image,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        visualize=False,
        feature_vector=True
    )
    
    return hog_features


def extract_hog_features_batch(images: List[Union[Image.Image, np.ndarray]], max_workers: int = 16) -> List[np.ndarray]:
    """
    Extract HOG features for a batch of images in parallel across multiple CPU processes.
    Using ProcessPoolExecutor to bypass the GIL for CPU-intensive HOG math.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(extract_hog_features, images))
    return results


class CLIPEmbedder:
    """Multimodal embedder (Supports CLIP and SigLIP)"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.CLIP_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        token = config.HF_TOKEN
        
        # Load in FP16 for H100 tensor core throughput (2x faster, no quality loss for embeddings)
        self.model = AutoModel.from_pretrained(
            self.model_name, token=token, dtype=torch.float16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name, token=token)
        self.model.eval()
        
        # Discover processor format for fast bypass (avoids slow NaFlex per-image analysis)
        self._init_fast_processor()
        
        print(f"Model loaded (FP16). Embedding dimension: {config.EMBEDDING_DIM}")
    
    def _init_fast_processor(self):
        """
        Probe the NaFlex processor with a single dummy image to discover:
        1. Output tensor keys and shapes
        2. Image normalization parameters (mean/std)
        3. Patch extraction parameters
        
        This lets us bypass the slow processor for fixed-size images.
        """
        try:
            # Get normalization params from processor config
            ip = self.processor.image_processor
            self._image_mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
            self._image_std = torch.tensor(ip.image_std, dtype=torch.float32).view(1, 3, 1, 1)
            
            # Probe with a dummy image to discover output format
            target_size = config.PREPROCESSING_TARGET_SIZE  # 224
            dummy = Image.new('RGB', (target_size, target_size), (128, 128, 128))
            ref = self.processor(images=dummy, return_tensors="pt")
            
            # Cache reference shapes and static tensors
            self._ref_shapes = {}
            self._ref_static = {}  # For tensors like spatial_shapes that are constant
            print(f"  NaFlex processor output format:")
            for k, v in ref.items():
                if isinstance(v, torch.Tensor):
                    self._ref_shapes[k] = v.shape
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                    # Store spatial_shapes as static (same for all same-size images)
                    if k == "spatial_shapes":
                        self._ref_static[k] = v.clone()
            
            # Determine patch size from the reference output
            # pixel_values should be [1, num_patches, patch_dim]
            pv_shape = ref["pixel_values"].shape
            if len(pv_shape) == 3:
                self._num_patches = pv_shape[1]
                self._patch_dim = pv_shape[2]
                # Infer patch_size: patch_dim = patch_size * patch_size * 3
                self._patch_size = int(np.sqrt(self._patch_dim / 3))
                print(f"    Detected: {self._num_patches} patches, dim={self._patch_dim}, patch_size={self._patch_size}")
            else:
                raise ValueError(f"Unexpected pixel_values shape: {pv_shape}")
            
            # Compute NaFlex target resolution from spatial_shapes
            # NaFlex resizes images to fit a specific grid (e.g. 16x16 = 256 patches)
            ref_ss = self._ref_static.get("spatial_shapes")
            if ref_ss is not None:
                grid_h = ref_ss[0, 0].item()
                grid_w = ref_ss[0, 1].item()
                self._naflex_h = grid_h * self._patch_size  # e.g. 16*16 = 256
                self._naflex_w = grid_w * self._patch_size
                print(f"    NaFlex target resolution: {self._naflex_h}x{self._naflex_w} (grid {grid_h}x{grid_w})")
            else:
                # Fallback: assume square based on num_patches
                grid = int(np.sqrt(self._num_patches))
                self._naflex_h = grid * self._patch_size
                self._naflex_w = grid * self._patch_size
            
            self._fast_processor_ready = True
            print(f"  Fast processor bypass: ENABLED")
        except Exception as e:
            self._fast_processor_ready = False
            print(f"  Fast processor bypass: DISABLED ({e})")
    
    def _fast_process_batch(self, batch_images: List[Image.Image]) -> dict:
        """
        Bypass the slow NaFlex processor for fixed-size images.
        
        NaFlex expects pixel_values as [B, num_patches, patch_dim] (pre-extracted patches),
        NOT [B, 3, H, W]. This function:
        1. Normalizes images (~0.01s)
        2. Extracts patches via unfold (~0.01s) 
        3. Creates attention mask + spatial_shapes (~0.001s)
        
        Total: ~0.05s vs ~30s with the NaFlex processor.
        """
        B = len(batch_images)
        ps = self._patch_size  # 16
        
        # Convert PIL images to numpy batch [B, H, W, 3] -> tensor [B, 3, H, W]
        batch_np = np.stack([np.asarray(img) for img in batch_images])  # uint8
        pixel_values = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0
        
        # Resize to NaFlex target resolution (e.g. 224x224 -> 256x256)
        # NaFlex selects a resolution bucket; for 224x224 input it uses 256x256
        if pixel_values.shape[2] != self._naflex_h or pixel_values.shape[3] != self._naflex_w:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                size=(self._naflex_h, self._naflex_w),
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize using cached mean/std (same as processor)
        pixel_values = (pixel_values - self._image_mean) / self._image_std
        
        # Extract patches: [B, 3, H, W] -> [B, num_patches, patch_dim]
        H, W = pixel_values.shape[2], pixel_values.shape[3]
        nH, nW = H // ps, W // ps
        
        patches = pixel_values.unfold(2, ps, ps).unfold(3, ps, ps)
        # Shape: [B, 3, nH, nW, ps, ps]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # Shape: [B, nH, nW, 3, ps, ps]
        patches = patches.reshape(B, nH * nW, -1)
        # Shape: [B, num_patches, patch_dim=768]
        patches = patches.half()  # FP16 to match model
        
        result = {"pixel_values": patches}
        
        # Attention mask: all patches are valid (fixed-size images)
        if "pixel_attention_mask" in self._ref_shapes:
            mask_shape = self._ref_shapes["pixel_attention_mask"]
            result["pixel_attention_mask"] = torch.ones(B, *mask_shape[1:], dtype=torch.long)
        
        # Spatial shapes: same for all images (fixed-size), expand from reference
        if "spatial_shapes" in self._ref_static:
            ref_ss = self._ref_static["spatial_shapes"]  # [1, ...]
            result["spatial_shapes"] = ref_ss.expand(B, *[-1] * (ref_ss.ndim - 1)).contiguous()
        
        # Pin memory for fast CPU->GPU transfer
        if self.device == "cuda":
            result = {k: v.pin_memory() for k, v in result.items()}
        
        return result
    
    @torch.no_grad()
    def embed_image(self, image: Union[str, Image.Image]) -> List[float]:
        """
        Generate embedding for a single image.
        
        If USE_HYBRID_EMBEDDINGS is True, creates a fused vector:
        V_final = Concat(sqrt(alpha) * V_siglip, sqrt(1-alpha) * V_hog)
        
        This allows weighted score fusion via cosine similarity.
        
        Args:
            image: PIL Image or path to image
        
        Returns:
            Embedding as list (768-dim for SigLIP only, 2532-dim for hybrid)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Get SigLIP features
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        
        # Handle both tensor and object outputs
        if hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0, :]
        else:
            features = outputs  # Already a tensor
        
        # Normalize SigLIP features
        siglip_vec = features / features.norm(p=2, dim=-1, keepdim=True)
        siglip_vec = siglip_vec.float().cpu().squeeze().numpy()
        
        # Create hybrid embedding if enabled
        if getattr(config, 'USE_HYBRID_EMBEDDINGS', False):
            # Extract and normalize HOG features
            hog_vec = extract_hog_features(image)
            hog_vec = normalize_vector(hog_vec)
            
            # Scaled concatenation for weighted score fusion
            # Score = alpha * SigLIP_sim + (1-alpha) * HOG_sim
            alpha = 1.0 - config.HOG_WEIGHT  # 0.8 for SigLIP
            beta = config.HOG_WEIGHT          # 0.2 for HOG
            
            # Scale vectors by sqrt of weights
            scaled_siglip = siglip_vec * np.sqrt(alpha)
            scaled_hog = hog_vec * np.sqrt(beta)
            
            # Concatenate to form hybrid vector
            hybrid_vec = np.concatenate([scaled_siglip, scaled_hog])
            
            return hybrid_vec.tolist()
        
        return siglip_vec.tolist()
    
    @torch.no_grad()
    def embed_images_batch(self, images: List[Union[str, Image.Image]], 
                           batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of PIL Images or paths
            batch_size: Batch size for processing
        
        Returns:
            List of embeddings (768-dim for SigLIP, 2532-dim for hybrid)
        """
        all_embeddings = []
        use_hybrid = getattr(config, 'USE_HYBRID_EMBEDDINGS', False)
        
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
            batch = images[i:i + batch_size]
            
            # Load images if paths
            pil_images = []
            for img in batch:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert('RGB'))
                else:
                    pil_images.append(img)
            
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_image_features(**inputs)
            
            # Handle both tensor and object outputs
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs
            
            # Normalize SigLIP features
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            siglip_batch = features.float().cpu().numpy()
            
            # Hybrid scaling factors
            alpha = 1.0 - config.HOG_WEIGHT
            beta = config.HOG_WEIGHT
            
            for j, siglip_vec in enumerate(siglip_batch):
                if use_hybrid:
                    # In a high-performance scenario, we'd pre-calculate HOGs for the entire chunk.
                    # For compatibility with this single-batch function, we still do it here
                    # but we'll optimize the caller (fast_h100_indexing.py) later.
                    hog_vec = extract_hog_features(pil_images[j])
                    hog_vec = normalize_vector(hog_vec)
                    
                    scaled_siglip = siglip_vec * np.sqrt(alpha)
                    scaled_hog = hog_vec * np.sqrt(beta)
                    hybrid_vec = np.concatenate([scaled_siglip, scaled_hog])
                    all_embeddings.append(hybrid_vec.tolist())
                else:
                    all_embeddings.append(siglip_vec.tolist())
        
        return all_embeddings
    
    def _process_batch_on_cpu(self, batch_images: List[Image.Image]) -> dict:
        """
        Run the NaFlex processor on CPU (the expensive part).
        Designed to be called from a background thread for pipelining.
        """
        inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
        # Pin memory for faster CPU->GPU transfer
        pinned = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                pinned[k] = v.pin_memory() if self.device == "cuda" else v
            else:
                pinned[k] = v
        return pinned

    @torch.no_grad()
    def embed_images_batch_hybrid_precalculated(self, 
                                               images: List[Image.Image], 
                                               hog_vectors: List[np.ndarray],
                                               batch_size: int = 512) -> List[List[float]]:
        """
        High-performance hybrid embedding.
        
        Uses fast processor bypass for fixed-size images:
        - NaFlex processor: ~30s per batch of 256 (per-image resolution analysis)
        - Fast bypass: ~0.05s per batch of 512 (direct tensor creation)
        
        Falls back to slow processor if fast path fails.
        """
        n = len(images)
        alpha = 1.0 - config.HOG_WEIGHT
        beta = config.HOG_WEIGHT
        sqrt_alpha = np.float32(np.sqrt(alpha))
        sqrt_beta = np.float32(np.sqrt(beta))
        
        total_batches = (n + batch_size - 1) // batch_size
        batch_ranges = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
        
        # Pre-allocate output array
        embed_dim = config.SIGLIP_DIM + config.HOG_DIM
        all_embeddings = np.empty((n, embed_dim), dtype=np.float32)
        
        # Vectorized HOG normalization for ALL vectors at once
        hog_array = np.stack(hog_vectors).astype(np.float32)
        hog_norms = np.linalg.norm(hog_array, axis=1, keepdims=True)
        hog_norms[hog_norms == 0] = 1.0
        hog_array /= hog_norms
        hog_array *= sqrt_beta  # Pre-scale
        
        use_fast = self._fast_processor_ready
        if use_fast:
            print(f"    Using FAST processor bypass (batch={batch_size})")
        else:
            print(f"    Using slow NaFlex processor (batch={batch_size})")
        
        pbar = tqdm(total=total_batches, desc="  GPU Embedding", leave=False)
        
        for start, end in batch_ranges:
            batch_images = images[start:end]
            
            # Process batch: fast bypass or slow NaFlex
            if use_fast:
                try:
                    inputs = self._fast_process_batch(batch_images)
                except Exception as e:
                    print(f"    Fast path failed ({e}), falling back to processor")
                    inputs = self._process_batch_on_cpu(batch_images)
            else:
                inputs = self._process_batch_on_cpu(batch_images)
            
            # GPU forward pass
            inputs_gpu = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs_gpu)
            
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            else:
                features = outputs
            
            # Normalize SigLIP on GPU, then move to CPU
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            siglip_batch = features.float().cpu().numpy()  # float32 for concatenation
            
            # Vectorized fusion: scale + concatenate entire batch at once
            siglip_scaled = siglip_batch * sqrt_alpha
            all_embeddings[start:end] = np.concatenate(
                [siglip_scaled, hog_array[start:end]], axis=1
            )
            
            pbar.update(1)
        
        pbar.close()
        
        return all_embeddings.tolist()

    @torch.no_grad()
    def embed_images_batch_with_preprocessing(self, image_paths: List[str], 
                                               batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for images with inline preprocessing.
        Preprocessing: grayscale -> binarize -> center -> resize 224x224 -> RGB
        
        This is more reliable than separate preprocessing step for large datasets.
        
        Args:
            image_paths: List of paths to images
            batch_size: Batch size for processing
        
        Returns:
            List of 768-dimensional embeddings
        """
        from preprocessing import preprocess_character_image
        
        all_embeddings = []
        total = len(image_paths)
        
        for i in tqdm(range(0, total, batch_size), desc="Preprocessing + Embedding"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Preprocess and load images
            pil_images = []
            for path in batch_paths:
                try:
                    preprocessed = preprocess_character_image(path)
                    pil_images.append(preprocessed)
                except Exception as e:
                    # Create blank image on error
                    blank = Image.new('RGB', (224, 224), color=(255, 255, 255))
                    pil_images.append(blank)
            
            # Generate embeddings
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_image_features(**inputs)
            
            # Handle both tensor and object outputs
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs
            
            # Normalize
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            all_embeddings.extend(features.float().cpu().tolist())
        
        return all_embeddings
    
    @torch.no_grad()
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text query.
        
        Args:
            text: Text string
        
        Returns:
            768-dimensional embedding as list
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        
        # Handle both tensor and object outputs
        if hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0, :]
        else:
            features = outputs
        
        # Normalize
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        
        return features.float().cpu().squeeze().tolist()
    
    @torch.no_grad()
    def embed_texts_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
        
        Returns:
            List of 768-dimensional embeddings
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i + batch_size]
            
            inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            
            # Handle both tensor and object outputs
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs
            
            # Normalize
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            all_embeddings.extend(features.float().cpu().tolist())
        
        return all_embeddings


if __name__ == "__main__":
    # Test embeddings
    embedder = CLIPEmbedder()
    
    # Test image embedding
    test_image = config.TRAIN_DIR / "1.jpg"
    if test_image.exists():
        img_emb = embedder.embed_image(str(test_image))
        print(f"Image embedding shape: {len(img_emb)}")
    
    # Test text embedding
    text_emb = embedder.embed_text("Malayalam handwriting")
    print(f"Text embedding shape: {len(text_emb)}")
