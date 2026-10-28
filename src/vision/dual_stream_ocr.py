"""
Dual-Stream Vision & Character Extraction Module
Stream A: SigLIP image embeddings matched to spreadsheet response embeddings via HERA retrieval
Stream B: Structural stroke localization for teacher review

Crucial Constraint: Strictly preserves children's orthographic errors (NO language-model autocorrect).
Performs actual OpenCV stroke segmentation and bounding-box localization on uploaded images.
"""

from typing import Dict, Any, Optional, List
import os
import cv2
import numpy as np
from PIL import Image

from src.vision.hera_recognizer import HeraRecognizer
from src.vision.hera_rag import HeraSigLIPWorkbookRAG
from src.vision.qwen2_vl import Qwen2VLMRecognizer


class DualStreamOCR:
    """
    Embeds handwriting with SigLIP, retrieves workbook-grounded response candidates,
    and keeps stroke localization available for teacher review.
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.hera = HeraRecognizer()
        self.hera_rag = HeraSigLIPWorkbookRAG()
        self.model_name = self.hera_rag.model_name

    def process_handwriting_image(
        self,
        image_bytes: bytes,
        output_dir: str = "app/static/uploads",
        filename_prefix: str = "scan",
        sheet_candidates: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Runs visual localization and target-aware SigLIP-to-workbook retrieval.
        """
        # Keep stroke localization for visual review; spelling retrieval is
        # performed by SigLIP against response candidates from the workbook.
        hera_res = self.hera.segment_and_recognize(
            image_bytes=image_bytes,
            output_dir=output_dir,
            filename_prefix=filename_prefix
        )

        if not hera_res.get("success"):
            return hera_res

        # The legacy overlay labels boxes with guesses from a font demo.
        # Retain localization but show neutral boxes instead of false labels.
        raw = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if raw is not None:
            boxed = raw.copy()
            for token in hera_res.get("tokens", []):
                box = token.get("box")
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    x, y, w, h = (int(value) for value in box)
                    cv2.rectangle(boxed, (x, y), (x + w, y + h), (16, 185, 129), 2)
            annotated_path = os.path.join(output_dir, f"{filename_prefix}_annotated.png")
            cv2.imwrite(annotated_path, boxed)

        detected_boxes_count = len(hera_res.get("tokens", []))
        
        # ── STREAM B: HERA Retrieval with VPP Spatial Prior ──
        retrieval = self.hera_rag.retrieve(
            image_bytes=image_bytes,
            sheet_candidates=sheet_candidates,
            detected_boxes_count=detected_boxes_count
        )
        if not retrieval.get("success"):
            # Test/gallery images are not a runtime dependency. Preserve image
            # upload and typed-response analysis if SigLIP cannot load.
            retrieval = {
                "success": True,
                "spelled_word": "",
                "candidates": [],
                "sheet_candidates": [],
                "sheet_candidate_count": 0,
                "matching_mode": retrieval.get("matching_mode", "SIGLIP_UNAVAILABLE"),
                "vision_similarity": None,
                "recognizer_mode": retrieval.get("error", "SigLIP unavailable; teacher transcription needed"),
                "autocorrect_applied": False,
            }

        spelled_word = retrieval.get("spelled_word", "")
        recognizer_mode = retrieval.get("recognizer_mode", "Dual-Stream HERA Vision")
        anomalies = self._detect_spatial_stroke_anomalies(spelled_word)

        hera_res.update({
            "spelled_word": spelled_word,
            "vision_confidence": None,
            "stream_a_hera_confidence": None,
            "stream_b_siglip_confidence": None,
            "recognizer_mode": recognizer_mode,
            "visual_candidates": retrieval.get("candidates", []),
            "sheet_candidates": retrieval.get("sheet_candidates", []),
            "sheet_candidate_count": retrieval.get("sheet_candidate_count", 0),
            "matching_mode": retrieval.get("matching_mode"),
            "vision_similarity": retrieval.get("vision_similarity"),
            "autocorrect_applied": False,
            "spatial_anomalies": anomalies
        })

        return hera_res

    def extract_verbatim_graphemes(
        self,
        image_bytes: Optional[bytes] = None,
        simulated_raw_transcription: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts verbatim Aksharas from a handwritten word image.
        Guarantees 0% dictionary autocorrection.
        """
        if image_bytes is not None:
            hera_res = self.process_handwriting_image(image_bytes)
            return {
                "spelled_word": hera_res.get("spelled_word", ""),
                "vision_confidence": hera_res.get("vision_confidence", 0.0),
                "stream_a_hera_confidence": hera_res.get("stream_a_hera_confidence", 0.0),
                "stream_b_siglip_confidence": hera_res.get("stream_b_siglip_confidence", 0.0),
                "autocorrect_applied": False,
                "tokens": hera_res.get("tokens", []),
                "visual_candidates": hera_res.get("visual_candidates", []),
                "vision_similarity": hera_res.get("vision_similarity"),
                "recognizer_mode": hera_res.get("recognizer_mode"),
                "spatial_anomalies": hera_res.get("spatial_anomalies", [])
            }

        raw_text = (simulated_raw_transcription or "").strip()
        return {
            "spelled_word": raw_text,
            "vision_confidence": None,
            "stream_a_hera_confidence": None,
            "stream_b_siglip_confidence": None,
            "autocorrect_applied": False,  # Strict guarantee
            "recognizer_mode": "manual text input",
            "spatial_anomalies": self._detect_spatial_stroke_anomalies(raw_text)
        }

    def _detect_spatial_stroke_anomalies(self, text: str) -> list:
        """Flags non-linear spatial irregularities (e.g. duplicated diacritics)."""
        anomalies = []
        if not text:
            return anomalies
            
        vowels = ["അ", "ആ", "ഇ", "ഈ", "ഉ", "ഊ", "ഋ", "എ", "ഏ", "ഐ", "ഒ", "ഓ", "ഔ"]
        consonants = ["ക", "ഖ", "ഗ", "ഘ", "ങ", "ച", "ഛ", "ജ", "ഝ", "ഞ", 
                      "ട", "ഠ", "ഡ", "ഢ", "ണ", "ത", "ഥ", "ദ", "ധ", "ന", 
                      "പ", "ഫ", "ബ", "ഭ", "മ", "യ", "ര", "ല", "വ", "ശ", "ഷ", "സ", "ഹ", "ള", "ഴ", "റ"]
        
        for i in range(len(text) - 1):
            if text[i] in consonants and text[i+1] in vowels:
                anomalies.append({
                    "type": "independent_vowel_after_consonant",
                    "position": i,
                    "tokens": text[i:i+2],
                    "clinical_note": "Child used primary vowel form instead of attached diacritic sign"
                })
        return anomalies
