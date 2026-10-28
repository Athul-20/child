"""HERA retrieval over workbook response candidates using shared SigLIP space."""
from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


def count_base_graphemes(word: str) -> int:
    """Counts base Akshara glyph bodies (vowels, consonants, chillus) excluding combining diacritics."""
    return sum(1 for ch in word if (0x0D05 <= ord(ch) <= 0x0D3A) or (0x0D7A <= ord(ch) <= 0x0D7F))


class HeraSigLIPWorkbookRAG:
    """Match an uploaded image embedding to this target's workbook responses.

    The gold images used in evaluation are not a runtime dependency. The
    fine-tuned Malayalam checkpoint/CTC head from the research plan is absent,
    so the public base SigLIP model provides suggestions requiring review.
    """

    # The HERA app uses this same multilingual SigLIP2 base model for its
    # available base-encoder path; a fine-tuned local checkpoint can override it.
    DEFAULT_MODEL = "google/siglip2-base-patch16-naflex"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("MALAYALAM_SIGLIP_MODEL", self.DEFAULT_MODEL)
        self.model = None
        self.processor = None
        self.torch = None
        self.device = None
        self.load_error: str | None = None
        self._candidate_embedding_cache: dict[tuple[str, ...], np.ndarray] = {}

    def _load_model(self) -> bool:
        if self.model is not None:
            return True
        if self.load_error:
            return False
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to(self.device).eval()
            except Exception:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            self.torch = torch
            return True
        except Exception as exc:
            self.load_error = f"SigLIP could not be loaded ({type(exc).__name__})."
            return False

    @staticmethod
    def _pooled_features(output):
        """Normalize Transformers 4/5 return variants to a pooled tensor."""
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if isinstance(output, (tuple, list)):
            # BaseModelOutputWithPooling tuple order is (last_hidden, pooled).
            if len(output) > 1:
                return output[1]
            if output:
                return output[0]
        return output

    def retrieve(self, image_bytes: bytes, sheet_candidates: list[dict[str, Any]] | None = None,
                 limit: int = 5, detected_boxes_count: int | None = None) -> dict[str, Any]:
        rows = [row for row in (sheet_candidates or [])
                if row.get("word") and not row.get("ambiguous")]
        if not rows:
            return {"success": False, "error": "No unambiguous response spellings for this target were found in the workbook.",
                    "matching_mode": "TARGET_HAS_NO_SHEET_RESPONSES"}
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return {"success": False, "error": "The uploaded file is not a readable image."}
        if not self._load_model():
            return {"success": False, "error": self.load_error or "SigLIP is unavailable.",
                    "matching_mode": "SIGLIP_UNAVAILABLE"}

        words = list(dict.fromkeys(str(row["word"]).strip() for row in rows))
        rows_by_word = {str(row["word"]).strip(): row for row in rows}

        try:
            image_inputs = self.processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items() if hasattr(v, "to")}
            with self.torch.inference_mode():
                image_output = self.model.get_image_features(**image_inputs)
                image_features = self._pooled_features(image_output).float()
                image_features = self.torch.nn.functional.normalize(image_features, dim=-1)

                text_features_cpu = self._candidate_embedding_cache.get(tuple(words))
                if text_features_cpu is None:
                    # Multi-template ensemble: bare word and targeted language prefixes
                    word_feature_list = []
                    for word in words:
                        prompts = [word, f"Malayalam: {word}", f"Malayalam word: {word}"]
                        text_inputs = self.processor(text=prompts, padding=True, return_tensors="pt")
                        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items() if hasattr(v, "to")}
                        text_output = self.model.get_text_features(**text_inputs)
                        tf = self._pooled_features(text_output).float()
                        tf = self.torch.nn.functional.normalize(tf, dim=-1)
                        mean_tf = self.torch.nn.functional.normalize(tf.mean(dim=0, keepdim=True), dim=-1)
                        word_feature_list.append(mean_tf.cpu())
                    text_features_cpu = self.torch.cat(word_feature_list, dim=0).numpy()
                    self._candidate_embedding_cache[tuple(words)] = text_features_cpu

                text_features = self.torch.from_numpy(text_features_cpu).to(self.device)
                raw_scores = (image_features @ text_features.T).squeeze(0).cpu().numpy()
                if raw_scores.ndim == 0:
                    raw_scores = np.array([float(raw_scores)])
        except Exception as exc:
            return {"success": False, "error": f"SigLIP could not embed this image ({type(exc).__name__}: {str(exc)[:180]}).",
                    "matching_mode": "SIGLIP_EMBEDDING_FAILED"}

        # Dual-Stream Spatial Prior: Fuse visual embedding with detected stroke contour clusters
        score_by_word = {}
        for word, score in zip(words, raw_scores):
            final_score = float(score)
            if detected_boxes_count and detected_boxes_count > 0:
                base_count = count_base_graphemes(word)
                diff = abs(base_count - detected_boxes_count)
                final_score += 0.05 if diff == 0 else (-0.03 * diff)
            score_by_word[word] = final_score

        row_by_word = rows_by_word
        ranked = sorted(words, key=lambda word: score_by_word[word], reverse=True)
        matches = [{**row_by_word[word], "similarity": round(score_by_word[word], 5),
                    "raw_vision_similarity": round(float(raw_score), 5),
                    "similarity_is_calibrated_confidence": False} for word, raw_score in zip(ranked[:limit], [raw_scores[words.index(w)] for w in ranked[:limit]])]
        best = matches[0]
        return {"success": True, "spelled_word": best["word"], "candidates": matches,
                "sheet_candidates": matches, "sheet_candidate_count": len(words),
                "matching_mode": "SIGLIP_IMAGE_TO_WORKBOOK_TEXT_RAG",
                "vision_similarity": best["similarity"],
                "recognizer_mode": f"Dual-Stream HERA Vision (SigLIP Multi-Template + Contour Spatial Prior over {len(words)} workbook responses)",
                "autocorrect_applied": False}
