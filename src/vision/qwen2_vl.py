"""
Qwen2-VL CPU-Optimized Malayalam Handwriting Transcriber.
Designed to run on standard CPU with low memory footprint and zero autocorrect bias.
"""

import os
from typing import Optional, Dict, Any
from io import BytesIO
from PIL import Image

class Qwen2VLMRecognizer:
    """
    CPU-optimized Qwen2-VL recognizer for Malayalam handwritten Aksharas.
    """
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        self.model_id = model_id or self.MODEL_ID
        self.device = device
        self.model = None
        self.processor = None
        self.load_error = None

    def load_model(self) -> bool:
        if self.model is not None:
            return True
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            print(f"Loading {self.model_id} on {self.device}...")
            # On CPU, use float32 or bfloat16 for fast execution
            dtype = torch.bfloat16 if torch.cuda.is_available() and self.device == "cuda" else torch.float32
            
            # Limit resolution so CPU inference takes < 2 seconds
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

    def transcribe(self, image_input: Any) -> Dict[str, Any]:
        """
        Transcribes handwritten Malayalam text with strict anti-autocorrect instructions.
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

            prompt = (
                "You are an expert Malayalam handwriting transcriber for a child clinical dyslexia tool. "
                "Transcribe the handwritten Malayalam characters exactly as written by the child. "
                "CRITICAL: Do not spell-check or autocorrect errors. "
                "If the child wrote full independent vowels instead of vowel signs or diacritics (e.g. അടഇ instead of അടി), "
                "transcribe the raw letters written: അടഇ. "
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
                    do_sample=False  # Greedy decoding for consistent verbatim transcription
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

            return {
                "success": True,
                "spelled_word": output_text,
                "model": self.model_id,
                "device": self.device
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Inference failed ({type(e).__name__}: {str(e)})",
                "spelled_word": ""
            }

