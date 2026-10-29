"""
Multi-Agent Orchestrator & Diagnostic Engine:
Orchestrates:
1. Error-Analysis Agent (HERA Retrieval-Only OCR + 3-tier G2P + PanPhon + 8,673-Row Gold RAG)
2. Insight Agent (Longitudinal memory + Theil-Sen + BKT + CUSUM + Critique-Revise Loop)
3. Intervention Planner Agent (RTI Tier recommendations + Pedagogical drill generation)
4. Teacher Verification Gate (Human-in-the-loop review)
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import unicodedata

from src.vision.dual_stream_ocr import DualStreamOCR
from src.phonology.three_tier_g2p import ThreeTierG2PEngine
from src.phonology.panphon_analyzer import PanPhonAnalyzer
from src.rag.gold_taxonomy_search import GoldTaxonomySearch
from src.stats.closed_loop_evaluator import ClosedLoopEvaluator


class DiagnosticOrchestrator:
    """
    End-to-End Orchestrator for Malayalam Child Handwriting Evaluation.
    """

    def __init__(
        self,
        taxonomy_path: str = "Malayalam_Data_With_Generated_IPA_ErrorsHandled_updated (1).xlsx",
        lexicon_path: str = "IPA- syllablewise (1) (1) (1).xlsx"
    ):
        self.ocr = DualStreamOCR()
        self.g2p = ThreeTierG2PEngine(lexicon_excel_path=lexicon_path)
        self.panphon = PanPhonAnalyzer()
        self.rag = GoldTaxonomySearch(excel_path=taxonomy_path)
        self.evaluator = ClosedLoopEvaluator()

    # ══════════════════════════════════════════════════════════════════════════
    # AGENT 1: ERROR-ANALYSIS AGENT
    # ══════════════════════════════════════════════════════════════════════════
    def run_error_analysis(
        self,
        target_word: str = "",
        image_bytes: Optional[bytes] = None,
        simulated_raw_handwriting: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a handwriting image or text input through:
        1. HERA visual exemplar retrieval
        2. 8,673-row clinical error dataset grounding
        3. 3-Tier G2P & 22-D PanPhon articulatory feature decomposition
        """
        target = unicodedata.normalize("NFC", (target_word or "").strip())
        sheet_candidates = self.rag.candidate_responses_for_target(target) if target else None
        
        # Step 1: Multimodal Qwen2-VL Vision Perception (0% autocorrect)
        ocr_res = self.ocr.extract_verbatim_graphemes(
            image_bytes=image_bytes,
            simulated_raw_transcription=simulated_raw_handwriting,
            target_word=target,
            sheet_candidates=sheet_candidates
        )
        spelled_word = unicodedata.normalize("NFC", ocr_res["spelled_word"].strip())

        # A known target is essential for a defensible probe comparison. Never
        # infer it from an error corpus or silently substitute a demo word.
        if not target:
            raise ValueError("Expected probe word is required for analysis.")
        if not spelled_word:
            raise ValueError("Child response is required for analysis.")

        # Prefer the exact pair's expert-labelled IPA when the taxonomy has it.
        # Otherwise use the word-to-IPA workbook and its deterministic fallback.
        exact_pair = self.rag.search_exact_pair(target, spelled_word)
        exact_pair_usable = bool(exact_pair and not exact_pair.get("ambiguous"))
        target_source = self.g2p.word_ipa_source(target)
        spelled_source = self.g2p.word_ipa_source(spelled_word)
        target_ipa = ((exact_pair or {}).get("real_word_ipa") if exact_pair_usable else None) or self.g2p.osur_g2p_tier1(target)
        spelled_ipa_primary = ((exact_pair or {}).get("spelled_ipa") if exact_pair_usable else None) or self.g2p.osur_g2p_tier1(spelled_word)
        target_ipa = target_ipa.replace(".", "").strip()
        spelled_ipa_primary = spelled_ipa_primary.replace(".", "").strip()
        expert_ipa = bool(exact_pair_usable and exact_pair.get("real_word_ipa") and exact_pair.get("spelled_ipa"))
        unique_lexicon_pair = "unique word-to-IPA" in target_source and "unique word-to-IPA" in spelled_source
        ipa_source = "exact expert taxonomy pair" if expert_ipa else ("unique word-to-IPA workbook entries" if unique_lexicon_pair else "deterministic G2P fallback; reference IPA unavailable or ambiguous")

        # Step 4: Tier 2 Deterministic Rule G2P
        spelled_ipa_rule = self.g2p.epitran_tier2(spelled_word)

        # Step 5: Cross-Validation Confidence Gate
        if expert_ipa:
            gate_res = {
                "status": "GOLD_PAIR",
                "edit_distance": self.g2p._levenshtein_distance(
                    spelled_ipa_primary.replace(".", ""), spelled_ipa_rule.replace(".", "")
                ),
                "confidence": "EXPERT-LABELLED PAIR",
                "verified_ipa": spelled_ipa_primary,
                "note": "IPA values loaded from the exact target/response row in the expert taxonomy."
            }
            verified_ipa = spelled_ipa_primary
        elif unique_lexicon_pair:
            gate_res = self.g2p.cross_validate(spelled_ipa_primary, spelled_ipa_rule)
            verified_ipa = gate_res["verified_ipa"]
        else:
            gate_res = {
                "status": "IPA_UNVERIFIED",
                "edit_distance": None,
                "confidence": "LOW",
                "verified_ipa": spelled_ipa_rule,
                "note": "No unique reference IPA is available for cross-validation; deterministic transliteration is shown for review."
            }
            verified_ipa = spelled_ipa_rule

        # Step 6: Identify character difference locus
        target_char, spelled_char = self._identify_diff_locus(target, spelled_word)

        # Step 7: Tier 3 PanPhon 22-D Articulatory Feature Decomposition
        wfd = self.panphon.calculate_word_wfd(target_ipa, verified_ipa)
        
        differing_features = []
        full_feature_table = []
        clinical_category = "Orthographic / Structural"
        severity = "NONE"

        if target_ipa != verified_ipa:
            t_clean = target_ipa.replace(".", "").replace(" ", "")
            s_clean = verified_ipa.replace(".", "").replace(" ", "")
            # Character-wise IPA comparison is only defensible when lengths
            # match. Insertions/deletions need phoneme alignment, so do not
            # invent a feature-level substitution for those cases.
            if len(t_clean) == len(s_clean):
                for tc, sc in zip(t_clean, s_clean):
                    if tc != sc:
                        pair_res = self.panphon.analyze_phoneme_pair(tc, sc)
                        differing_features = pair_res["differing_features"]
                        full_feature_table = pair_res.get("full_feature_table", [])
                        clinical_category = pair_res["clinical_category"]
                        severity = pair_res["severity"]
                        break

        # Step 8: Gold Database Precedent RAG Search over 8,673 Records
        rag_res = self.rag.search_precedent(
            target_word=target,
            spelled_word=spelled_word,
            target_char=target_char,
            spelled_char=spelled_char
        )

        # Step 9: Diagnostic Classification Synthesis
        is_error = (target != spelled_word)
        if not is_error:
            error_type = "CORRECT"
            dimension = "None"
            feedback = "Accurate spelling and phonological execution."
            row_id = None
        elif rag_res["matches"] and rag_res["matches"][0]["similarity_score"] >= 0.80:
            top_match = rag_res["matches"][0]
            error_type = top_match.get("error_type", "Phonological Error" if differing_features else "Orthographic Error")
            dimension = top_match.get("dimension", clinical_category)
            feedback = top_match.get("feedback", "")
            row_id = top_match.get("row_id")
        else:
            # Autonomous clinical synthesis based on PanPhon & G2P features
            error_type = "Phonological Error" if differing_features else "Orthographic Error"
            dimension = clinical_category
            row_id = None

            if differing_features:
                feedback = (
                    f"Child substituted {spelled_char or spelled_word} in place of {target_char or target_word}. "
                    f"PanPhon subsegmental analysis confirms a {clinical_category} divergence with "
                    f"differing features: {list(set(differing_features))} (WFD: {wfd})."
                )
            elif ocr_res.get("spatial_anomalies"):
                feedback = ocr_res["spatial_anomalies"][0]["clinical_note"]
                dimension = "Multi-linearity (Diacritic Placement)"
            else:
                feedback = f"Orthographic Akshara substitution between '{target_char or target}' and '{spelled_char or spelled_word}'."

        return {
            "target_word": target,
            "spelled_word": spelled_word,
            "target_ipa": target_ipa,
            "spelled_ipa": verified_ipa,
            "g2p_confidence": gate_res["confidence"],
            "cross_val_status": gate_res["status"],
            "is_error": is_error,
            "error_type": error_type,
            "dimension": dimension,
            "feedback": feedback,
            "differing_articulatory_features": list(set(differing_features)),
            "full_feature_table": full_feature_table,
            "weighted_feature_distance": wfd,
            "articulatory_severity": severity,
            "rag_precedent_id": row_id,
            "hera_tokens": ocr_res.get("tokens", []),
            "recognizer_mode": ocr_res.get("recognizer_mode", "manual text input"),
            "g2p_mode": "lexicon lookup with deterministic rule fallback; OSUR neural checkpoint unavailable",
            "ipa_source": ipa_source,
            "taxonomy_pair_status": "AMBIGUOUS" if exact_pair and exact_pair.get("ambiguous") else ("EXACT" if exact_pair else "NO_EXACT_PAIR")
        }

    @staticmethod
    def _identify_diff_locus(w1: str, w2: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify the first differing Malayalam akshara without splitting signs."""
        def aksharas(text: str) -> List[str]:
            units: List[str] = []
            join_next = False
            for char in text:
                is_mark = unicodedata.category(char).startswith("M")
                if not units or (not is_mark and not join_next):
                    units.append(char)
                else:
                    units[-1] += char
                join_next = char == "്"
            return units

        left, right = aksharas(w1), aksharas(w2)
        for c1, c2 in zip(left, right):
            if c1 != c2:
                return c1, c2
        if len(left) > len(right):
            return "".join(left[len(right):]), None
        elif len(right) > len(left):
            return None, "".join(right[len(left):])
        return None, None

    # ══════════════════════════════════════════════════════════════════════════
    # AGENT 2: LONGITUDINAL EVALUATION AGENT
    # ══════════════════════════════════════════════════════════════════════════
    def run_longitudinal_evaluation(
        self,
        student_id: str,
        dimension: str,
        session_history: List[Dict[str, Any]],
        new_session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes longitudinal analysis with self-verifying critique-revise loop.
        """
        return self.evaluator.evaluate_session_delta(
            student_id=student_id,
            dimension=dimension,
            historical_sessions=session_history,
            current_session_result=new_session
        )
