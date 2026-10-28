"""
Gold Database RAG Precedent Search Module.
Loads and indexes the 8,673-row expert Malayalam clinical error taxonomy:
'Malayalam_Data_With_Generated_IPA_ErrorsHandled_updated (1).xlsx'

Provides bidirectional grounding:
- (Real Word, Spelled Word) -> Exact Clinical Diagnosis
- Spelled Word alone (from HERA image recognition) -> Target Word Recovery & Error Classification
- Isolated Akshara / Substituted Letter -> Authentic Clinical Rationale from Corpus
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import os
import re
import unicodedata


class GoldTaxonomySearch:
    """
    RAG Precedent Search over the 8,673-row clinical error dataset.
    """

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.df = None
        self._exact_index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._spelled_index: Dict[str, List[Dict[str, Any]]] = {}
        self._target_index: Dict[str, List[Dict[str, Any]]] = {}
        self._all_entries: List[Dict[str, Any]] = []
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Taxonomy Excel not found at: {self.excel_path}")
        
        self.df = pd.read_excel(self.excel_path)
        
        for idx, row in self.df.iterrows():
            real_w = unicodedata.normalize("NFC", str(row['Real Word']).strip()) if pd.notna(row['Real Word']) else ""
            spelled_w = unicodedata.normalize("NFC", str(row['Spelled Word']).strip()) if pd.notna(row['Spelled Word']) else ""
            dim = str(row.get('Dimension', '')).strip() if pd.notna(row.get('Dimension')) else ""
            sub = str(row.get('Error Sub-type', '')).strip() if pd.notna(row.get('Error Sub-type')) else ""
            fb = str(row.get('Error feedback', '')).strip() if pd.notna(row.get('Error feedback')) else ""
            pos = str(row.get('Error Position', '')).strip() if pd.notna(row.get('Error Position')) else ""
            err_type = str(row.get('Error ', '')).strip() if pd.notna(row.get('Error ')) else ""

            if not err_type:
                err_type = "Phonological Error" if "Phon" in dim else "Orthographic Error"

            entry = {
                "row_id": int(idx),
                "real_word": real_w,
                "spelled_word": spelled_w,
                "real_word_ipa": str(row.get('Real word IPA', '')).strip() if pd.notna(row.get('Real word IPA')) else "",
                "spelled_ipa": str(row.get('Spelled IPA', '')).strip() if pd.notna(row.get('Spelled IPA')) else "",
                "error_type": err_type,
                "dimension": dim,
                "subtype": sub,
                "feedback": fb,
                "position": pos,
                "similarity_score": 1.0
            }
            
            self._all_entries.append(entry)

            # Exact pair index
            if real_w and spelled_w:
                self._exact_index.setdefault((real_w, spelled_w), []).append(entry)

            # Spelled word index (for image recognition when target is unknown)
            if spelled_w:
                if spelled_w not in self._spelled_index:
                    self._spelled_index[spelled_w] = []
                self._spelled_index[spelled_w].append(entry)

            # Target word index
            if real_w:
                if real_w not in self._target_index:
                    self._target_index[real_w] = []
                self._target_index[real_w].append(entry)

    def search_by_spelled_word(self, spelled_word: str) -> Optional[Dict[str, Any]]:
        """
        Reverse lookup: Given a child's handwritten word (e.g. from HERA OCR),
        recovers the ground-truth target word and clinical classification from the 8,673 corpus.
        """
        spelled = spelled_word.strip()
        if not spelled:
            return None

        # 1. Exact match in spelled index
        if spelled in self._spelled_index:
            entry = self._spelled_index[spelled][0]
            entry_copy = dict(entry)
            entry_copy["similarity_score"] = 1.0
            entry_copy["match_type"] = "HERA_EXACT_SPELLED_MATCH"
            return entry_copy

        # 2. Check if spelled_word is an isolated Akshara (e.g. "പ", "ക", "ഉ", "ല")
        # Search the clinical corpus for entries where that letter was used as a substitution
        if len(spelled) <= 2:
            for entry in self._all_entries:
                fb = entry["feedback"]
                if f"using {spelled} instead of" in fb or f"using {spelled} " in fb:
                    entry_copy = dict(entry)
                    entry_copy["similarity_score"] = 0.92
                    entry_copy["match_type"] = "CLINICAL_AKSHARA_SUBSTITUTION"
                    return entry_copy

        # 3. Substring / Root match within the 8,673 dataset
        for s_w, entries in self._spelled_index.items():
            if spelled in s_w or s_w in spelled:
                entry_copy = dict(entries[0])
                entry_copy["similarity_score"] = 0.82
                entry_copy["match_type"] = "CORPUS_PARTIAL_MATCH"
                return entry_copy

        return None

    def search_exact_pair(self, target_word: str, spelled_word: str) -> Optional[Dict[str, Any]]:
        """Return the workbook's ground-truth row only for an exact word pair."""
        target = unicodedata.normalize("NFC", target_word.strip())
        spelled = unicodedata.normalize("NFC", spelled_word.strip())
        entries = self._exact_index.get((target, spelled), [])
        if not entries:
            return None

        def norm(value: Any) -> str:
            return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

        label_signatures = {
            (norm(e["error_type"]), norm(e["dimension"]), norm(e["subtype"]))
            for e in entries
        }
        real_ipas = {e["real_word_ipa"].strip() for e in entries if e["real_word_ipa"]}
        spelled_ipas = {e["spelled_ipa"].strip() for e in entries if e["spelled_ipa"]}
        if len(label_signatures) > 1 or len(real_ipas) > 1 or len(spelled_ipas) > 1:
            return {"ambiguous": True, "match_count": len(entries)}

        # Prefer a duplicate row with complete IPA fields when labels agree.
        selected = next((e for e in entries if e["real_word_ipa"] and e["spelled_ipa"]), entries[0])
        result = dict(selected)
        result["ambiguous"] = False
        result["match_count"] = len(entries)
        return result

    def candidate_responses_for_target(self, target_word: str) -> List[Dict[str, Any]]:
        """List only spellings and reference labels documented for this probe."""
        target = unicodedata.normalize("NFC", (target_word or "").strip())
        responses: List[Dict[str, Any]] = []
        for (real_word, spelled_word), entries in self._exact_index.items():
            if unicodedata.normalize("NFC", real_word) != target:
                continue
            # Keep the expected answer out of the error-candidate set so a
            # correct response cannot win merely because it is the prompt.
            if spelled_word == target:
                continue
            pair = self.search_exact_pair(target, spelled_word) or {}
            representative = entries[0]
            responses.append({
                "target_word": target,
                "word": spelled_word,
                "error_type": representative.get("error_type", ""),
                "dimension": representative.get("dimension", ""),
                "subtype": representative.get("subtype", ""),
                "ambiguous": bool(pair.get("ambiguous")),
                "row_id": representative.get("row_id")
            })
        return responses

    def search_precedent(
        self,
        target_word: str,
        spelled_word: str,
        target_char: Optional[str] = None,
        spelled_char: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves the authentic clinical precedent from the 8,673 corpus.
        Crucial requirement: Never returns feedback for characters the child did not write.
        """
        target = target_word.strip()
        spelled = spelled_word.strip()

        # Only an exact word pair is safe to present as a precedent. Character
        # overlap alone can attach an unrelated clinical label to this sample.
        exact_match = self.search_exact_pair(target, spelled)
        if exact_match and exact_match.get("ambiguous"):
            return {"match_type": "AMBIGUOUS_EXACT_PAIR", "similarity_score": 0.0, "matches": []}
        if exact_match:
            return {
                "match_type": "EXACT",
                "similarity_score": 1.0,
                "matches": [exact_match]
            }

        # 2. Target word is empty -> reverse search by spelled word
        if not target and spelled:
            res = self.search_by_spelled_word(spelled)
            if res:
                return {
                    "match_type": res.get("match_type", "EXACT_SPELLED"),
                    "similarity_score": res.get("similarity_score", 0.95),
                    "matches": [res]
                }

        # Without an exact pair, report no precedent and use transparent
        # deterministic analysis instead of borrowing another learner's case.
        return {
            "match_type": "SYNTHESIZED",
            "similarity_score": 0.0,
            "matches": []
        }
