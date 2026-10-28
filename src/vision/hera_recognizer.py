"""
HERA: Handwritten Error Recognition Architecture (Pillar 1)
Error-Preserving Malayalam Akshara Recognition via Visual Exemplar Retrieval.

Methodology from Doctoral Framework (Slide 8):
1. Hierarchical 3-tier segmentation:
   - Tier 1: Vertical Projection Profile (VPP) for cluster & column valleys
   - Tier 2: Canny / external contour extraction for connected strokes
   - Tier 3: Width & aspect-ratio fallback for merged conjuncts / diacritics
2. Hybrid Descriptor:
   - Histogram of Oriented Gradients (HOG) + topological stroke contour zoning (density, aspect, symmetry, loops)
3. Retrieval-Only Recognition:
   - Nearest-neighbour exemplar prototype matching with inverse-distance confidence
   - Zero language-model bias: strictly preserves verbatim child misspellings
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple, Optional


class HeraRecognizer:
    """
    Retrieval-only Malayalam handwriting recognizer implementing the HERA architecture.
    """

    def __init__(self, font_path: str = r"C:\Windows\Fonts\Nirmala.ttc"):
        self.font_path = font_path
        self.image_size = (64, 64)
        self.exemplar_bank: List[Dict[str, Any]] = []
        self._build_exemplar_bank()

    def _build_exemplar_bank(self):
        """
        Builds the reference prototype descriptor bank for Malayalam graphemes:
        - Base consonants, vowels, chillus, matras, and conjuncts.
        """
        if not os.path.exists(self.font_path):
            print(f"Warning: Font {self.font_path} not found. HERA will use fallback descriptors.")
            return

        try:
            font = ImageFont.truetype(self.font_path, 40)
        except Exception as e:
            print(f"Error loading font: {e}")
            return

        # Comprehensive Malayalam Character Inventory
        characters = [
            # Consonants
            "ക", "ഖ", "ഗ", "ഘ", "ങ",
            "ച", "ഛ", "ജ", "ഝ", "ഞ",
            "ട", "ഠ", "ഡ", "ഢ", "ണ",
            "ത", "ഥ", "ദ", "ധ", "ന",
            "പ", "ഫ", "ബ", "ഭ", "മ",
            "യ", "ര", "ല", "വ", "ശ", "ഷ", "സ", "ഹ", "ള", "ഴ", "റ",
            # Vowels
            "അ", "ആ", "ഇ", "ഈ", "ഉ", "ഊ", "ഋ", "എ", "ഏ", "ഐ", "ഒ", "ഓ", "ഔ",
            # Chillus
            "ൽ", "ൾ", "ൻ", "ൺ", "ർ", "ൿ",
            # Attached Diacritics / Matras
            "ാ", "ി", "ീ", "ു", "ൂ", "ൃ", "െ", "േ", "ൈ", "ൊ", "ോ", "ൌ", "്", "ം",
            # Frequent Learner Conjuncts / Combinations
            "ക്ക", "ങ്ങ", "ച്ച", "ഞ്ഞ", "ട്ട", "ണ്ണ", "ത്ത", "ന്ദ", "ന്ന", "പ്പ", 
            "മ്മ", "യ്യ", "ല്ല", "വ്വ", "ള്ള", "ക്ഷ", "ജ്ഞ"
        ]

        for ch in characters:
            # Render character image
            img = Image.new("L", (80, 80), 255)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = max(4, (80 - tw) // 2)
            ty = max(4, (80 - th) // 2)
            draw.text((tx, ty), ch, fill=0, font=font)

            np_img = np.array(img)
            # Threshold
            _, bin_img = cv2.threshold(np_img, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Crop to ink
            coords = cv2.findNonZero(bin_img)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                cropped = bin_img[y:y+h, x:x+w]
            else:
                cropped = bin_img

            desc = self._compute_hybrid_descriptor(cropped)
            if desc is not None:
                self.exemplar_bank.append({
                    "char": ch,
                    "descriptor": desc,
                    "aspect_ratio": round(w / max(1, h), 3) if coords is not None else 1.0
                })

    def _compute_hybrid_descriptor(self, binary_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Computes the normalized hybrid feature vector:
        HOG (Gradient orientations) + Topological zoning features.
        """
        if binary_crop is None or binary_crop.size == 0:
            return None

        # Resize to fixed 64x64 while preserving aspect ratio
        h, w = binary_crop.shape
        scale = min(56.0 / max(1, w), 56.0 / max(1, h))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(binary_crop, (nw, nh), interpolation=cv2.INTER_AREA)

        padded = np.zeros((64, 64), dtype=np.uint8)
        px = (64 - nw) // 2
        py = (64 - nh) // 2
        padded[py:py+nh, px:px+nw] = resized

        # 1. HOG descriptor (8 bins, 8x8 cells, 16x16 blocks)
        try:
            hog = cv2.HOGDescriptor(
                _winSize=(64, 64),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=8
            )
            hog_feats = hog.compute(padded).flatten()
        except Exception:
            hog_feats = np.zeros(1568, dtype=np.float32)

        # 2. Topological Zoning Features
        total_pixels = 64 * 64
        ink_pixels = np.count_nonzero(padded)
        density = float(ink_pixels / max(1, total_pixels))
        
        # Quadrant density ratios (top-half, bottom-half, left-half, right-half)
        top_half = np.count_nonzero(padded[0:32, :])
        bottom_half = np.count_nonzero(padded[32:64, :])
        left_half = np.count_nonzero(padded[:, 0:32])
        right_half = np.count_nonzero(padded[:, 32:64])

        v_sym = float(top_half / max(1, ink_pixels))
        h_sym = float(left_half / max(1, ink_pixels))

        # Euler characteristic / number of loops
        contours, hierarchy = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = 0
        if hierarchy is not None:
            for h_entry in hierarchy[0]:
                if h_entry[3] != -1:  # Has a parent contour -> is a hole
                    holes += 1

        topological_feats = np.array([
            density,
            v_sym,
            h_sym,
            float(w / max(1, h)),
            float(holes) * 0.25
        ], dtype=np.float32)

        # Concatenate into hybrid vector and L2-normalize
        hybrid = np.concatenate([hog_feats * 0.85, topological_feats * 2.5])
        norm = np.linalg.norm(hybrid)
        if norm > 1e-6:
            hybrid = hybrid / norm

        return hybrid

    def segment_and_recognize(
        self,
        image_bytes: bytes,
        output_dir: str = "app/static/uploads",
        filename_prefix: str = "hera_scan"
    ) -> Dict[str, Any]:
        """
        Full HERA Pipeline:
        1. Hierarchical 3-Tier Segmentation
        2. Visual Feature Extraction
        3. Nearest-Neighbour Retrieval against Exemplar Bank
        4. Verbatim Error-Preserving Transcription
        """
        os.makedirs(output_dir, exist_ok=True)

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "error": "Invalid image format"}

        h, w, c = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ── TIER 1: Vertical Projection Profile (VPP) ──
        vpp = np.sum(thresh, axis=0) / 255.0
        active_cols = np.where(vpp > 2)[0]

        # ── TIER 2: Contour Segmentation ──
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Filter noise
            if bw * bh > 70 and bw < w * 0.98 and bh < h * 0.98:
                raw_boxes.append((bx, by, bw, bh))

        # ── TIER 3: Width / Conjunct Fallback (VPP Valley Guided) ──
        refined_boxes = []
        for bx, by, bw, bh in raw_boxes:
            crop = thresh[by:by+bh, bx:bx+bw]
            vpp = np.sum(crop, axis=0) / 255.0
            mid_start = int(bw * 0.25)
            mid_end = int(bw * 0.75)
            split_applied = False
            # Only split if box is wide AND has a genuine vertical projection valley (no strokes connecting)
            if mid_end > mid_start and (bw / max(1, bh)) > 2.2:
                mid_vpp = vpp[mid_start:mid_end]
                min_val = np.min(mid_vpp)
                if min_val <= 1.0:
                    split_x = mid_start + int(np.argmin(mid_vpp))
                    refined_boxes.append((bx, by, split_x, bh))
                    refined_boxes.append((bx + split_x, by, bw - split_x, bh))
                    split_applied = True
            if not split_applied:
                refined_boxes.append((bx, by, bw, bh))

        # Sort left-to-right (Malayalam reading order)
        refined_boxes.sort(key=lambda b: b[0])

        # If no contours found, take central crop
        if not refined_boxes:
            refined_boxes = [(0, 0, w, h)]

        # ── RETRIEVAL-ONLY RECOGNITION (HERA Metric Learning) ──
        recognized_tokens = []
        annotated_img = img.copy()

        for idx, (bx, by, bw, bh) in enumerate(refined_boxes, 1):
            crop = thresh[by:by+bh, bx:bx+bw]
            crop_desc = self._compute_hybrid_descriptor(crop)

            best_char = "?"
            best_sim = -1.0
            second_char = ""
            second_sim = -1.0

            if crop_desc is not None and self.exemplar_bank:
                for ex in self.exemplar_bank:
                    sim = float(np.dot(crop_desc, ex["descriptor"]))
                    if sim > best_sim:
                        second_sim = best_sim
                        second_char = best_char
                        best_sim = sim
                        best_char = ex["char"]
                    elif sim > second_sim:
                        second_sim = sim
                        second_char = ex["char"]

            confidence = round(max(0.10, min(0.99, (best_sim + 1.0) / 2.0)), 3)
            
            recognized_tokens.append({
                "box_id": f"A{idx}",
                "box": [bx, by, bw, bh],
                "char": best_char,
                "confidence": confidence,
                "top_alternative": second_char if second_sim > 0.4 else None,
                "aspect_ratio": round(bw / max(1, bh), 2)
            })

            # Draw green bounding box & character label on annotated image
            cv2.rectangle(annotated_img, (bx, by), (bx + bw, by + bh), (16, 185, 129), 2)
            cv2.putText(
                annotated_img,
                f"A{idx}: {best_char}",
                (bx, max(18, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (6, 182, 212),
                2
            )

        # Assemble raw learner word from recognized tokens
        spelled_word = "".join(t["char"] for t in recognized_tokens if t["char"] != "?")

        # Save visualization files
        raw_path = os.path.join(output_dir, f"{filename_prefix}_raw.png")
        annotated_path = os.path.join(output_dir, f"{filename_prefix}_annotated.png")
        cv2.imwrite(raw_path, img)
        cv2.imwrite(annotated_path, annotated_img)

        return {
            "success": True,
            "spelled_word": spelled_word,
            "tokens": recognized_tokens,
            "segmented_boxes_count": len(refined_boxes),
            "dimensions": f"{w}x{h}",
            "stroke_density": round(float(np.count_nonzero(thresh) / max(1, w * h)), 4),
            "raw_image_url": f"/static/uploads/{filename_prefix}_raw.png",
            "annotated_image_url": f"/static/uploads/{filename_prefix}_annotated.png",
            "autocorrect_applied": False,
            "architecture": "HERA (Retrieval-Only Metric Learning)"
        }

