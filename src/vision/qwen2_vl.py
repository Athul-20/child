"""
Qwen2-VL Multimodal Vision Recognizer for Malayalam Handwriting.
Specialized for distinguishing cursive, touching, or continuous Akshara strokes
without dictionary autocorrection bias.
"""

import os
from typing import Optional, Dict, Any, List
from io import BytesIO
from PIL import Image


class Qwen2VLMRecognizer:
    """
    Qwen2-VL 2B Vision-Language Recognizer for handwritten Malayalam Aksharas.
    Leverages visual transformer attention across touching and continuous character strokes.
    """
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        self.model_id = model_id or self.MODEL_ID
        if device is not None:
            self.device = device
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        self.model = None
        self.processor = None
        self.load_error = None

    def load_model(self) -> bool:
        if self.model is not None:
            return True
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            print(f"Loading {self.model_id} on {self.device} (offline local cache first)...")
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            # Try loading directly from local cache (0 network calls, no re-downloading)
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    min_pixels=256 * 28 * 28,
                    max_pixels=512 * 28 * 28,
                    local_files_only=True
                )
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map=self.device,
                    low_cpu_mem_usage=True,
                    local_files_only=True
                )
            except Exception:
                # Fallback only if model files are not yet in local cache
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    min_pixels=256 * 28 * 28,
                    max_pixels=512 * 28 * 28
                )
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map=self.device,
                    low_cpu_mem_usage=True
                )
            return True
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {str(e)}"
            print(f"Failed to load {self.model_id}: {self.load_error}")
            return False

    def transcribe(
        self,
        image_input: Any,
        target_word: Optional[str] = None,
        candidates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Transcribes handwritten Malayalam text, handling continuous or touching letters.
        Maintains strict anti-autocorrect fidelity to observe actual child misspellings.
        """
        if not self.load_model():
            return {
                "success": False,
                "error": self.load_error or "Qwen2-VL failed to load.",
                "spelled_word": ""
            }

        try:
            import torch
            if isinstance(image_input, (bytes, bytearray)):
                image = Image.open(BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                return {"success": False, "error": "Invalid image input format", "spelled_word": ""}

            cand_list: List[str] = []
            if candidates:
                for c in candidates:
                    clean_c = c.strip() if isinstance(c, str) else ""
                    if clean_c and clean_c not in cand_list:
                        cand_list.append(clean_c)

            if target_word:
                clean_target = target_word.strip()
                if clean_target and clean_target not in cand_list:
                    cand_list.insert(0, clean_target)

            if cand_list and len(cand_list) > 1:
                cand_str = ", ".join(f"'{c}'" for c in cand_list)
                prompt = (
                    "You are an expert clinical Malayalam handwriting examiner for child literacy screening.\n"
                    + (f"The expected target word was '{target_word}'.\n" if target_word else "")
                    + "Look very carefully at the child's handwriting in the image from left to right, even if characters touch, slant, or have no spaces between them.\n"
                    "Differentiate closely related letters based on stroke curvature:\n"
                    "- 'ഗ' (ga): has a smooth round arch with two downward legs and NO loop or notch.\n"
                    "- 'ഖ' (kha): has an initial loop or notch before the arch.\n"
                    "- 'ഘ' (gha): has a bottom loop curve.\n"
                    "- 'ക' (ka): has crossed loop strokes.\n"
                    f"Which word did the child write? Candidates: [{cand_str}].\n"
                    "Output ONLY the exact Malayalam word and nothing else."
                )
            else:
                prompt = (
                    "You are an expert clinical Malayalam handwriting examiner for child literacy screening.\n"
                    + (f"The expected target word was '{target_word}'.\n" if target_word else "")
                    + "Look very carefully at the child's handwriting in the image from left to right, even if characters touch, slant, or have no spaces between them.\n"
                    "Transcribe the raw handwritten Malayalam characters exactly as written by the child.\n"
                    "CRITICAL: Do not spell-check, autocorrect, or substitute characters.\n"
                    "Output ONLY the raw Malayalam text characters and nothing else."
                )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False  # Deterministic greedy decoding
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

            # Clean output
            cleaned = output_text.strip().strip("'\"` \n\r\t")
            if cand_list:
                for c in cand_list:
                    if c == cleaned or f"'{c}'" in output_text or f'"{c}"' in output_text:
                        cleaned = c
                        break

            return {
                "success": True,
                "spelled_word": cleaned,
                "raw_output": output_text,
                "model": self.model_id,
                "device": self.device,
                "candidates": cand_list
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Inference failed ({type(e).__name__}: {str(e)})",
                "spelled_word": ""
            }
