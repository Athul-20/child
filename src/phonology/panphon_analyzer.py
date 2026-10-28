"""
PanPhon Articulatory Feature Decomposition & Distance Engine.
Decomposes IPA segments into 22-dimensional articulatory feature vectors
and computes subsegmental phonetic distances.
"""

from typing import Dict, Any, List, Optional
import os


class PanPhonAnalyzer:
    """
    Tier 3 Articulatory Phonetics Engine.
    Converts IPA symbols into 22-dimensional articulatory vectors
    and calculates Weighted Feature Distance (WFD).
    """

    def __init__(self):
        self.ft = None
        self.dst = None
        self._init_panphon()

    def _init_panphon(self):
        """Initializes panphon with UTF-8 support."""
        try:
            import panphon
            import panphon.distance
            self.ft = panphon.FeatureTable()
            self.dst = panphon.distance.Distance()
        except Exception as e:
            print(f"Warning: PanPhon init note: {e}")

    def analyze_phoneme_pair(self, target_phone: str, spelled_phone: str) -> Dict[str, Any]:
        """
        Compares two IPA phones across 22 articulatory dimensions.
        Returns both the divergent features and the complete 22-D comparative feature table.
        """
        if self.ft is None:
            return self._fallback_comparison(target_phone, spelled_phone)

        try:
            t_vecs = self.ft.word_to_vector_list(target_phone, numeric=True)
            s_vecs = self.ft.word_to_vector_list(spelled_phone, numeric=True)

            differing = []
            full_table = []

            if t_vecs and s_vecs:
                t_vec = t_vecs[0]
                s_vec = s_vecs[0]
                names = self.ft.names
                for name, t_val, s_val in zip(names, t_vec, s_vec):
                    t_sym = "+" if t_val > 0 else ("-" if t_val < 0 else "0")
                    s_sym = "+" if s_val > 0 else ("-" if s_val < 0 else "0")
                    is_diff = (t_val != s_val)

                    entry = {
                        "feature": name,
                        "target_val": t_sym,
                        "spelled_val": s_sym,
                        "is_divergent": is_diff
                    }
                    full_table.append(entry)

                    if is_diff:
                        differing.append(entry)

            wfd = self._normalized_sequence_distance(t_vecs, s_vecs)
            feature_count = len(differing)

            severity = "LOW" if feature_count <= 2 else ("MEDIUM" if feature_count <= 5 else "HIGH")

            return {
                "target_phone": target_phone,
                "spelled_phone": spelled_phone,
                "feature_distance": feature_count,
                "weighted_feature_distance": round(float(wfd), 4),
                "differing_features": [d["feature"] for d in differing],
                "detailed_diffs": differing,
                "full_feature_table": full_table,
                "severity": severity,
                "clinical_category": self._classify_articulatory_dimension(differing)
            }
        except Exception as e:
            return self._fallback_comparison(target_phone, spelled_phone)

    def calculate_word_wfd(self, target_ipa: str, spelled_ipa: str) -> float:
        """Return normalized feature-edit distance in the range [0, 1]."""
        if target_ipa == spelled_ipa:
            return 0.0
        if self.ft is None:
            return 1.0
        try:
            t_vecs = self.ft.word_to_vector_list(target_ipa, numeric=True)
            s_vecs = self.ft.word_to_vector_list(spelled_ipa, numeric=True)
            return round(float(self._normalized_sequence_distance(t_vecs, s_vecs)), 4)
        except Exception:
            return 1.0

    @staticmethod
    def _normalized_sequence_distance(target_vectors, spelled_vectors) -> float:
        """Levenshtein-style distance with 22-D feature cost, normalized to 0..1."""
        n, m = len(target_vectors), len(spelled_vectors)
        if n == 0 and m == 0:
            return 0.0
        if n == 0 or m == 0:
            return 1.0

        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = float(i)
        for j in range(1, m + 1):
            dp[0][j] = float(j)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left, right = target_vectors[i - 1], spelled_vectors[j - 1]
                width = max(len(left), len(right), 1)
                paired = sum(abs(float(a) - float(b)) for a, b in zip(left, right))
                cost = paired / (2.0 * width)
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,
                    dp[i][j - 1] + 1.0,
                    dp[i - 1][j - 1] + cost
                )
        return min(1.0, dp[n][m] / max(n, m))

    @staticmethod
    def _classify_articulatory_dimension(differing: List[Dict[str, str]]) -> str:
        """Maps differing binary features to speech pathology dimensions."""
        feats = {d["feature"] for d in differing}
        if "voi" in feats or "voice" in feats:
            return "Voicing Distinction (Phonetic Similarity)"
        if "retroflex" in feats or "ant" in feats:
            return "Place of Articulation (Dental/Alveolar vs. Retroflex Liquid/Nasal)"
        if "cont" in feats or "delrel" in feats:
            return "Manner of Articulation (Tap vs. Trill / Stop vs. Fricative)"
        if "sg" in feats or "spread_glottis" in feats:
            return "Aspiration Distinction"
        if "long" in feats or "tense" in feats:
            return "Vowel Length (Short vs Long Matra)"
        return "Composite Articulatory Substitution"

    @staticmethod
    def _fallback_comparison(target: str, spelled: str) -> Dict[str, Any]:
        differs = (target != spelled)
        return {
            "target_phone": target,
            "spelled_phone": spelled,
            "feature_distance": 1 if differs else 0,
            "weighted_feature_distance": 0.25 if differs else 0.0,
            "differing_features": ["phonetic_substitution"] if differs else [],
            "detailed_diffs": [],
            "full_feature_table": [],
            "severity": "LOW" if differs else "NONE",
            "clinical_category": "Phonetic Similarity"
        }
