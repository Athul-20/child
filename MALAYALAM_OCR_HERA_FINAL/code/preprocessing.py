"""
Shared Image Preprocessing for Malayalam Character OCR

This module provides consistent preprocessing for both:
1. Indexing (training images) - applied before generating embeddings for ChromaDB
2. Query (uploaded images) - applied before similarity search

Consistent preprocessing ensures embeddings are comparable across indexed and query images.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Union

# Try to import TARGET_SIZE from config, fallback to 224
try:
    from config import PREPROCESSING_TARGET_SIZE as TARGET_SIZE
except ImportError:
    TARGET_SIZE = 224


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert any image mode to RGB, compositing RGBA onto white."""
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # Always operate on RGBA to have an alpha channel for the mask
        rgba_img = image.convert('RGBA')
        bg = Image.new('RGB', rgba_img.size, (255, 255, 255))
        bg.paste(rgba_img, mask=rgba_img.split()[3])
        return bg
    return image.convert('RGB')


def preprocess_character_image_fast(image: Union[str, Image.Image, np.ndarray], return_grayscale: bool = False) -> Image.Image:
    """
    MINIMAL preprocessing for bulk indexing - just resize, no heavy ops.
    """
    # Load image
    if isinstance(image, str):
        pil_img = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_img = image.convert('RGB')
    else:
        pil_img = Image.fromarray(image).convert('RGB')
    
    # Just resize to target size
    pil_img = pil_img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    
    if return_grayscale:
        return pil_img.convert('L')
    return pil_img


def preprocess_character_image(image: Union[str, Image.Image, np.ndarray], return_grayscale: bool = False) -> Image.Image:
    """
    Preprocess a character image for embedding generation.
    
    This function normalizes images by:
    1. Converting to grayscale
    2. Applying Gaussian blur for noise reduction
    3. Binarizing with Sauvola threshold
    4. Finding and centering the character content
    5. Resizing to TARGET_SIZE x TARGET_SIZE with aspect ratio preservation
    6. Converting back to RGB for CLIP (unless return_grayscale=True)
    
    Args:
        image: PIL Image, path to image, or numpy array
        return_grayscale: If True, return grayscale image instead of RGB
        
    Returns:
        Preprocessed PIL Image (RGB or grayscale, TARGET_SIZE x TARGET_SIZE)
    """
    # Load image based on input type
    if isinstance(image, str):
        img_np = cv2.imread(image)
        if img_np is None:
            pil_img = Image.open(image).convert('RGB')
            img_np = np.array(pil_img)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert('RGB'))
    else:
        img_np = image
    
    # 1. Convert to grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()
    
    # 2. Gaussian Blur (reduce noise)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 3. Sauvola Binarization (Better for documents/handwriting with artifacts)
    window_size = 25
    k = 0.3
    r = 128
    
    # Float conversion for calculation
    img_float = gray.astype(float)
    mean = cv2.blur(img_float, (window_size, window_size))
    mean_sq = cv2.blur(img_float**2, (window_size, window_size))
    std_dev = np.sqrt(mean_sq - mean**2)
    
    threshold = mean * (1 + k * ((std_dev / r) - 1))
    
    binary = np.zeros_like(gray)
    binary[img_float < threshold] = 255 # Text becomes white, background black
    
    # 4. Smart Cleaning (Connected Component Analysis)
    # Goal: Remove the "Box" (frame) and small noise, keep the main Character
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    clean_binary = np.zeros_like(binary)
    
    # Image dimensions
    h_img, w_img = binary.shape
    img_center = (w_img // 2, h_img // 2)
    total_area = h_img * w_img
    
    candidates = []
    
    for i in range(1, num_labels): # Skip 0 (background)
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        
        # Filter A: Remove small noise
        if area < 50: 
            continue
            
        # Filter B: Remove Frame/Box
        # Check if component touches the edges
        touches_edge = (x <= 1) or (y <= 1) or (x+w >= w_img-1) or (y+h >= h_img-1)
        
        # If it's large (>50% of image) and touches edges, it's likely a border artifact
        if area > total_area * 0.5 and touches_edge:
            continue
            
        # Calculate distance from center
        dist_from_center = np.sqrt((cx - img_center[0])**2 + (cy - img_center[1])**2)
        candidates.append((i, area, dist_from_center))
        
    # Heuristic: Keep the single "best" component (Character)
    if candidates:
        # Sort by Area (largest first)
        candidates.sort(key=lambda c: c[1], reverse=True)
        
        # Keep the top candidate
        best_label = candidates[0][0]
        clean_binary[labels == best_label] = 255
        
    binary = clean_binary
    
    # 5. Morphological Operations (Thicken strokes)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # 6. Centering & Resizing (Standard logic)
    # Find bounding box of the cleaned character
    coords = cv2.findNonZero(binary)
    if coords is not None and len(coords) > 0:
        x, y, w, h = cv2.boundingRect(coords)
        # Crop to content with small margin
        margin = 2
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(binary.shape[1], x + w + margin)
        y2 = min(binary.shape[0], y + h + margin)
        binary_cropped = binary[y1:y2, x1:x2]
    else:
        binary_cropped = binary
    
    # Get dimensions
    h, w = binary_cropped.shape
    if h == 0 or w == 0:
        # Empty image - return blank
        result = np.ones((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) * 255
        if return_grayscale:
            return Image.fromarray(result, mode='L')
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(result_rgb)
    
    # Calculate scaling factor to fit in TARGET_SIZE
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h) * 0.85  # 85% to leave margin
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    # Resize
    resized = cv2.resize(binary_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create white background canvas
    result = np.ones((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) * 255
    
    # Invert so character is black on white (standard for CLIP/SigLIP)
    resized_inv = 255 - resized
    
    # Center the character
    x_offset = (TARGET_SIZE - new_w) // 2
    y_offset = (TARGET_SIZE - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_inv
    
    # Return grayscale if requested (for visualization)
    if return_grayscale:
        return Image.fromarray(result, mode='L')
    
    # Convert to RGB for CLIP
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(result_rgb)


def format_for_siglip(image: Union[Image.Image, np.ndarray], return_grayscale: bool = False) -> Image.Image:
    """
    Format an ALREADY SEGMENTED and cleaned image for the SigLIP model.
    This safely pads and resizes the image WITHOUT deleting disjoint components
    or artificially thickening strokes.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert('RGB'))
    else:
        img_np = image.copy()
        
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
        
    # Binarize with simple Otsu (since it's already clean)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find bounding box of all ink
    coords = cv2.findNonZero(binary)
    if coords is not None and len(coords) > 0:
        x, y, w, h = cv2.boundingRect(coords)
        margin = 2
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(binary.shape[1], x + w + margin)
        y2 = min(binary.shape[0], y + h + margin)
        binary_cropped = binary[y1:y2, x1:x2]
    else:
        binary_cropped = binary
        
    h, w = binary_cropped.shape
    if h == 0 or w == 0:
        result = np.ones((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) * 255
        return Image.fromarray(result, mode='L') if return_grayscale else Image.fromarray(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))
        
    # Calculate scaling factor to fit in TARGET_SIZE
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h) * 0.85  # 85% to leave margin
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    resized = cv2.resize(binary_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    result = np.ones((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) * 255
    resized_inv = 255 - resized
    
    x_offset = (TARGET_SIZE - new_w) // 2
    y_offset = (TARGET_SIZE - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_inv
    
    if return_grayscale:
        return Image.fromarray(result, mode='L')
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))


def get_binary_visualization(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Get the binary/preprocessed version of an image for visualization.
    
    This shows exactly what the model sees after preprocessing.
    
    Args:
        image: Input image
        
    Returns:
        Grayscale PIL Image showing the binarized, cleaned character
    """
    return preprocess_character_image(image, return_grayscale=True)


def preprocess_word_image(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Preprocess a WORD image (keeps all characters, not just the largest).
    
    Unlike preprocess_character_image which keeps only the largest component,
    this function keeps ALL meaningful components for word-level processing.
    
    Steps:
    1. Convert to grayscale
    2. Apply adaptive thresholding (Sauvola)
    3. Remove small noise but keep ALL character-sized components
    4. Remove border artifacts
    5. Return clean binary image (same size as input)
    
    Args:
        image: Input image
        
    Returns:
        Clean binary PIL Image (grayscale, same size)
    """
    # Load image
    if isinstance(image, str):
        img_np = cv2.imread(image)
        if img_np is None:
            pil_img = Image.open(image).convert('RGB')
            img_np = np.array(pil_img)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert('RGB'))
    else:
        img_np = image.copy() if hasattr(image, 'copy') else image
    
    # Convert to grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()
    
    h_img, w_img = gray.shape
    
    # 1. CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=8, templateWindowSize=7, searchWindowSize=21)
    
    # 3. Adaptive thresholding (Sauvola-style)
    # Use adaptive to handle uneven lighting
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Text white, background black
        blockSize=25, C=12
    )
    
    # 4. Clean using Connected Components - but keep ALL meaningful components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    clean_binary = np.zeros_like(binary)
    total_area = h_img * w_img
    
    # Calculate dynamic thresholds based on image size
    min_area = max(30, total_area // 2000)  # Minimum component area
    max_area = total_area * 0.6  # Maximum (to filter border boxes)
    
    # Collect valid component areas for median-based filtering
    valid_areas = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # Skip tiny noise
        if area < min_area:
            continue
            
        # Skip components that span almost the entire image (likely borders)
        if area > max_area:
            continue
            
        # Skip components touching all 4 edges (border box)
        touches_left = x <= 1
        touches_right = x + w >= w_img - 1
        touches_top = y <= 1
        touches_bottom = y + h >= h_img - 1
        
        if touches_left and touches_right and touches_top and touches_bottom:
            continue
            
        # Skip very elongated horizontal lines (likely underlines)
        aspect = w / max(h, 1)
        if aspect > 15 and h < 10:  # Very wide and thin
            continue
            
        valid_areas.append((i, area))
    
    if not valid_areas:
        # If nothing valid, return original with basic threshold
        _, simple_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return Image.fromarray(255 - simple_binary, mode='L')
    
    # Calculate median area for additional filtering
    areas_only = [a[1] for a in valid_areas]
    median_area = np.median(areas_only)
    
    # Keep components that are at least 10% of median area (removes tiny specks)
    min_relative = median_area * 0.10
    
    for label_id, area in valid_areas:
        if area >= min_relative:
            clean_binary[labels == label_id] = 255
    
    # 5. Morphological cleanup - close small gaps
    kernel = np.ones((2, 2), np.uint8)
    clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, kernel)
    
    # 6. Ensure PURE BINARY output (only 0 and 255)
    _, clean_binary = cv2.threshold(clean_binary, 127, 255, cv2.THRESH_BINARY)
    
    # Invert: black text on white background
    result = 255 - clean_binary
    
    # Guarantee pure binary
    result = np.where(result > 127, 255, 0).astype(np.uint8)
    
    return Image.fromarray(result, mode='L')


def get_word_binary_visualization(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Get clean binary visualization for a WORD image.
    Shows all characters, not just the largest one.
    """
    return preprocess_word_image(image)


def binarize_image(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Binarize an image using Sauvola thresholding.
    
    This is useful for preprocessing word images before segmentation.
    Unlike preprocess_character_image, this does NOT center/resize.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        
    Returns:
        Binarized PIL Image (grayscale, same size as input)
    """
    # Load image
    if isinstance(image, str):
        img_np = cv2.imread(image)
        if img_np is None:
            pil_img = Image.open(image).convert('RGB')
            img_np = np.array(pil_img)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert('RGB'))
    else:
        img_np = image.copy() if hasattr(image, 'copy') else image
    
    # Convert to grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # For high-contrast images (typical of PNG sources), close small gaps first
    contrast = int(gray.max()) - int(gray.min())
    if contrast > 180:
        kernel = np.ones((2, 2), np.uint8)
        blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    
    # Sauvola binarization
    window_size = 25
    k = 0.3
    r = 128
    
    img_float = blurred.astype(float)
    mean = cv2.blur(img_float, (window_size, window_size))
    mean_sq = cv2.blur(img_float**2, (window_size, window_size))
    std_dev = np.sqrt(np.maximum(mean_sq - mean**2, 0))  # Avoid negative sqrt
    
    threshold = mean * (1 + k * ((std_dev / r) - 1))
    
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[img_float < threshold] = 255  # Text white, background black
    
    # Invert to black text on white background
    binary_inv = 255 - binary
    
    return Image.fromarray(binary_inv, mode='L')


def _safe_preprocess_single(img_path):
    """
    Module-level wrapper for preprocessing that handles errors gracefully.
    """
    try:
        return preprocess_character_image(img_path)
    except Exception as e:
        # Return blank image on error
        blank = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255
        return Image.fromarray(blank)


def preprocess_images_batch(images: list, show_progress: bool = False, num_workers: int = None) -> list:
    """
    Preprocess a batch of images using parallel threading.
    
    Args:
        images: List of image paths
        show_progress: Whether to show progress bar
        num_workers: Number of parallel workers (default: CPU count * 2)
        
    Returns:
        List of preprocessed PIL Images
    """
    from concurrent.futures import ThreadPoolExecutor
    import os
    
    if num_workers is None:
        # Use more threads since this is I/O bound
        num_workers = min(32, (os.cpu_count() or 4) * 4)
    
    total = len(images)
    
    # For small batches, process sequentially
    if total < 50:
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(images, desc="Preprocessing images")
        else:
            iterator = images
        
        return [_safe_preprocess_single(img) for img in iterator]
    
    # Use ThreadPoolExecutor for large batches (faster for I/O)
    print(f"Using {num_workers} threads for parallel preprocessing...")
    
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="Preprocessing images")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map preserves order
        for result in executor.map(_safe_preprocess_single, images):
            results.append(result)
            if show_progress:
                pbar.update(1)
    
    if show_progress:
        pbar.close()
    
    return results


if __name__ == "__main__":
    # Test preprocessing
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"Testing preprocessing on: {test_path}")
        
        # Load original
        original = Image.open(test_path)
        print(f"Original size: {original.size}")
        
        # Preprocess
        preprocessed = preprocess_character_image(test_path)
        print(f"Preprocessed size: {preprocessed.size}")
        
        # Save for comparison
        preprocessed.save("test_preprocessed.png")
        print("Saved preprocessed image to: test_preprocessed.png")
        
        # Test shape features if --shape flag is provided
        if "--shape" in sys.argv:
            print("\n=== Shape Features ===")
            features = extract_all_shape_features(test_path)
            print(f"Hu Moments: {features['hu_moments']}")
            print(f"Contour: {features['contour']}")
            print(f"Skeleton: {features['skeleton']}")
    else:
        print("Usage: python preprocessing.py <image_path> [--shape]")


# =============================================================================
# SHAPE-BASED FEATURE EXTRACTION
# =============================================================================
# These features capture geometric structure for shape-based retrieval

from typing import Dict, List, Tuple, Any, Optional
from scipy import ndimage

# Try to import skimage, provide fallback if not available
try:
    from skimage.morphology import skeletonize, thin
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def preprocess_for_shape(image: Union[Image.Image, np.ndarray]) -> Tuple[np.ndarray, list]:
    """
    Preprocess image for shape feature extraction.
    
    Returns:
        Tuple of (binary_image, contours)
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    elif isinstance(image, str):
        img_np = np.array(Image.open(image).convert('RGB'))
    else:
        img_np = image.copy()
    
    # Convert to grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Denoise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Binarize with Otsu
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return binary, contours


# =============================================================================
# Hu Moments - Rotation/Scale/Translation Invariant
# =============================================================================

def extract_hu_moments(image: Union[Image.Image, np.ndarray, str]) -> Dict[str, float]:
    """
    Extract 7 Hu Moments from image.
    
    Hu Moments are invariant to translation, scale, and rotation,
    making them ideal for shape matching regardless of orientation.
    
    Returns:
        Dict with hu_1 through hu_7 (log-transformed for numerical stability)
    """
    binary, contours = preprocess_for_shape(image)
    
    if not contours:
        return {f'hu_{i+1}': 0.0 for i in range(7)}
    
    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform for numerical stability (handle zeros)
    hu_log = {}
    for i, h in enumerate(hu_moments):
        if h != 0:
            hu_log[f'hu_{i+1}'] = -np.sign(h) * np.log10(abs(h))
        else:
            hu_log[f'hu_{i+1}'] = 0.0
    
    return hu_log


# =============================================================================
# Contour-Based Shape Descriptors
# =============================================================================

def extract_contour_features(image: Union[Image.Image, np.ndarray, str]) -> Dict[str, float]:
    """
    Extract contour-based shape features.
    
    Features:
    - area: Contour area in pixels
    - perimeter: Contour perimeter length
    - circularity: 4π * area / perimeter² (1 = perfect circle)
    - solidity: area / convex_hull_area (how "solid" the shape is)
    - extent: area / bounding_rect_area (fill ratio)
    - aspect_ratio: width / height of bounding rectangle
    - compactness: perimeter² / area
    - convexity: convex_hull_perimeter / perimeter
    """
    binary, contours = preprocess_for_shape(image)
    
    if not contours:
        return {
            'area': 0, 'perimeter': 0, 'circularity': 0,
            'solidity': 0, 'extent': 0, 'aspect_ratio': 0,
            'compactness': 0, 'convexity': 0
        }
    
    # Combine all contours for overall shape
    all_points = np.vstack(contours)
    
    # Calculate area from binary image (more accurate)
    area = np.sum(binary > 0)
    
    # Perimeter from largest contour
    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    
    # Circularity
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 0
    
    # Convex hull
    hull = cv2.convexHull(all_points)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    
    # Solidity
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(all_points)
    rect_area = w * h
    
    # Extent
    extent = area / rect_area if rect_area > 0 else 0
    
    # Aspect ratio
    aspect_ratio = w / h if h > 0 else 0
    
    # Compactness
    compactness = (perimeter ** 2) / area if area > 0 else 0
    
    # Convexity
    convexity = hull_perimeter / perimeter if perimeter > 0 else 0
    
    return {
        'area': float(area),
        'perimeter': round(perimeter, 2),
        'circularity': round(circularity, 4),
        'solidity': round(solidity, 4),
        'extent': round(extent, 4),
        'aspect_ratio': round(aspect_ratio, 4),
        'compactness': round(compactness, 4),
        'convexity': round(convexity, 4)
    }


# =============================================================================
# Skeleton/Stroke Features
# =============================================================================

def extract_skeleton_features(image: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
    """
    Extract skeleton-based stroke features.
    
    Features:
    - stroke_length: Total length of skeleton
    - num_endpoints: Number of stroke endpoints
    - num_junctions: Number of junction points (where strokes meet)
    - branch_count: Number of branches in skeleton
    - stroke_width_avg: Average stroke width
    """
    binary, _ = preprocess_for_shape(image)
    
    # Normalize binary to 0-1 for skeletonize
    binary_norm = (binary > 0).astype(np.uint8)
    
    # Extract skeleton
    if SKIMAGE_AVAILABLE:
        skeleton = skeletonize(binary_norm).astype(np.uint8)
    else:
        # Fallback: use OpenCV thinning
        skeleton = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else binary_norm
    
    # Stroke length (number of skeleton pixels)
    stroke_length = np.sum(skeleton)
    
    if stroke_length == 0:
        return {
            'stroke_length': 0,
            'num_endpoints': 0,
            'num_junctions': 0,
            'branch_count': 0,
            'stroke_width_avg': 0
        }
    
    # Find endpoints and junctions using convolution
    # 3x3 kernel neighborhood count
    kernel = np.ones((3, 3), np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel) * skeleton
    
    # Endpoints: pixels with exactly 2 neighbors (itself + 1 other)
    endpoints = np.sum(neighbors == 2)
    
    # Junctions: pixels with 4+ neighbors (itself + 3+ others)
    junctions = np.sum(neighbors >= 4)
    
    # Branch count using connected components on skeleton
    num_labels, _ = cv2.connectedComponents(skeleton.astype(np.uint8))
    branch_count = num_labels - 1  # Subtract background
    
    # Average stroke width
    # Distance transform on binary, sample at skeleton points
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skeleton_distances = dist_transform[skeleton > 0]
    stroke_width_avg = np.mean(skeleton_distances) * 2 if len(skeleton_distances) > 0 else 0
    
    return {
        'stroke_length': int(stroke_length),
        'num_endpoints': int(endpoints),
        'num_junctions': int(junctions),
        'branch_count': int(branch_count),
        'stroke_width_avg': round(float(stroke_width_avg), 2)
    }


# =============================================================================
# Fourier Descriptors (Contour Frequency)
# =============================================================================

def extract_fourier_descriptors(image: Union[Image.Image, np.ndarray, str], num_descriptors: int = 10) -> List[float]:
    """
    Extract Fourier Descriptors from contour.
    
    Fourier Descriptors represent the contour shape in frequency domain,
    capturing both coarse and fine shape details.
    
    Args:
        image: Input image
        num_descriptors: Number of descriptors to return
        
    Returns:
        List of normalized Fourier descriptor magnitudes
    """
    binary, contours = preprocess_for_shape(image)
    
    if not contours:
        return [0.0] * num_descriptors
    
    # Use largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Reshape to complex numbers (x + jy)
    contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]
    
    # Apply FFT
    fft = np.fft.fft(contour_complex)
    
    # Normalize by DC component (makes scale invariant)
    if abs(fft[0]) > 0:
        fft_normalized = fft / abs(fft[0])
    else:
        fft_normalized = fft
    
    # Take magnitudes (makes rotation invariant)
    magnitudes = np.abs(fft_normalized)
    
    # Return first N descriptors (skip DC component)
    descriptors = magnitudes[1:num_descriptors+1].tolist()
    
    # Pad if contour has fewer points
    while len(descriptors) < num_descriptors:
        descriptors.append(0.0)
    
    return [round(d, 6) for d in descriptors]


# =============================================================================
# Shape Histogram (Angular Distribution)
# =============================================================================

def extract_shape_histogram(image: Union[Image.Image, np.ndarray, str], num_bins: int = 8) -> Dict[str, Any]:
    """
    Extract shape histogram based on angular distribution from centroid.
    
    Divides the shape into angular bins around the centroid and
    measures the distribution of pixels, capturing directional shape info.
    
    Returns:
        Dict with histogram bins and shape metrics
    """
    binary, _ = preprocess_for_shape(image)
    
    if np.sum(binary) == 0:
        return {
            'histogram': [0] * num_bins,
            'dominant_direction': 0,
            'direction_uniformity': 0
        }
    
    # Find centroid
    moments = cv2.moments(binary)
    if moments['m00'] == 0:
        return {
            'histogram': [0] * num_bins,
            'dominant_direction': 0,
            'direction_uniformity': 0
        }
    
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    
    # Get coordinates of all foreground pixels
    y_coords, x_coords = np.where(binary > 0)
    
    # Calculate angles from centroid
    angles = np.arctan2(y_coords - cy, x_coords - cx)
    
    # Convert to 0-2π range
    angles = (angles + np.pi) % (2 * np.pi)
    
    # Create histogram
    hist, _ = np.histogram(angles, bins=num_bins, range=(0, 2 * np.pi))
    
    # Normalize
    hist_normalized = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    
    # Dominant direction (bin with most pixels)
    dominant_bin = np.argmax(hist_normalized)
    dominant_direction = (dominant_bin + 0.5) * (360 / num_bins)  # Degrees
    
    # Direction uniformity (entropy-like measure)
    hist_nonzero = hist_normalized[hist_normalized > 0]
    if len(hist_nonzero) > 0:
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
        max_entropy = np.log2(num_bins)
        uniformity = entropy / max_entropy if max_entropy > 0 else 0
    else:
        uniformity = 0
    
    return {
        'histogram': [round(h, 4) for h in hist_normalized.tolist()],
        'dominant_direction': round(dominant_direction, 2),
        'direction_uniformity': round(uniformity, 4)
    }


# =============================================================================
# Combined Shape Features for Retrieval
# =============================================================================

def extract_all_shape_features(image: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
    """
    Extract all shape features for retrieval.
    
    Combines all feature extractors into a single comprehensive dict.
    
    Returns:
        Dict with all shape features
    """
    features = {}
    
    # Hu Moments (7 values)
    features['hu_moments'] = extract_hu_moments(image)
    
    # Contour features (8 values)
    features['contour'] = extract_contour_features(image)
    
    # Skeleton features (5 values)
    features['skeleton'] = extract_skeleton_features(image)
    
    # Fourier descriptors (10 values)
    features['fourier'] = extract_fourier_descriptors(image, num_descriptors=10)
    
    # Shape histogram (8 bins + 2 metrics)
    features['histogram'] = extract_shape_histogram(image, num_bins=8)
    
    return features


def extract_shape_vector(image: Union[Image.Image, np.ndarray, str]) -> List[float]:
    """
    Extract a flat shape feature vector for similarity matching.
    
    This creates a single vector that can be compared using distance metrics.
    
    Returns:
        List of floats (fixed length feature vector)
    """
    features = extract_all_shape_features(image)
    
    vector = []
    
    # Hu Moments (7)
    for i in range(1, 8):
        vector.append(features['hu_moments'].get(f'hu_{i}', 0))
    
    # Contour features (8)
    for key in ['area', 'perimeter', 'circularity', 'solidity', 
                'extent', 'aspect_ratio', 'compactness', 'convexity']:
        val = features['contour'].get(key, 0)
        # Normalize area and perimeter
        if key == 'area':
            val = np.log1p(val) / 10  # Log scale
        elif key == 'perimeter':
            val = np.log1p(val) / 10
        vector.append(val)
    
    # Skeleton features (5)
    for key in ['stroke_length', 'num_endpoints', 'num_junctions', 
                'branch_count', 'stroke_width_avg']:
        val = features['skeleton'].get(key, 0)
        if key == 'stroke_length':
            val = np.log1p(val) / 10
        vector.append(val)
    
    # Fourier descriptors (10)
    vector.extend(features['fourier'])
    
    # Shape histogram (8)
    vector.extend(features['histogram']['histogram'])
    
    return [round(v, 6) for v in vector]


def shape_similarity(features1: List[float], features2: List[float]) -> float:
    """
    Calculate shape similarity between two feature vectors.
    
    Returns:
        Similarity score (0-1, higher = more similar)
    """
    if len(features1) != len(features2):
        return 0.0
    
    # Euclidean distance
    distance = np.sqrt(np.sum((np.array(features1) - np.array(features2)) ** 2))
    
    # Convert to similarity (inverse distance, normalized)
    similarity = 1 / (1 + distance)
    
    return round(similarity, 4)


def extract_shape_metadata_for_indexing(image: Union[Image.Image, str]) -> Dict[str, Any]:
    """
    Extract minimal shape metadata suitable for vector DB indexing.
    
    Returns only key features that are useful for filtering/matching.
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    contour = extract_contour_features(image)
    skeleton = extract_skeleton_features(image)
    hu = extract_hu_moments(image)
    
    return {
        # Contour
        'shape_circularity': contour['circularity'],
        'shape_solidity': contour['solidity'],
        'shape_aspect_ratio': contour['aspect_ratio'],
        'shape_extent': contour['extent'],
        # Skeleton
        'shape_endpoints': skeleton['num_endpoints'],
        'shape_junctions': skeleton['num_junctions'],
        # Hu (first 3 are most discriminative)
        'shape_hu1': hu['hu_1'],
        'shape_hu2': hu['hu_2'],
        'shape_hu3': hu['hu_3'],
    }
