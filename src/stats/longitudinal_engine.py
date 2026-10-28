"""
Deterministic Longitudinal Statistical & Psychometric Engines:
1. Theil-Sen Robust Estimator (Outlier-resistant learning velocity, breakdown point 29.3%)
2. Bayesian Knowledge Tracing (BKT 4-parameter latent mastery model)
3. CUSUM Sequential Control Chart (Change-point & acute regression alert)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.stats import theilslopes


class LongitudinalStatsEngine:
    """
    Deterministic statistical services for child handwriting tracking.
    Guarantees that agents never guess or hallucinate numerical learning trends.
    """

    @staticmethod
    def theil_sen_slope(
        sessions: List[int],
        error_rates: List[float],
        alpha: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculates non-parametric Theil-Sen slope across assessment sessions.
        
        Args:
            sessions: List of sequential session numbers (e.g. [1, 2, 3, 4, 5])
            error_rates: List of error proportions (e.g. [0.45, 0.40, 0.50, 0.25, 0.15])
            alpha: Confidence level
        """
        if len(sessions) < 2 or len(error_rates) < 2:
            return {
                "slope": 0.0,
                "intercept": error_rates[0] if error_rates else 0.0,
                "lower_ci": 0.0,
                "upper_ci": 0.0,
                "verdict": "INSUFFICIENT_DATA",
                "learning_velocity": 0.0
            }

        res = theilslopes(error_rates, sessions, alpha=alpha)
        slope, intercept, low_slope, high_slope = res.slope, res.intercept, res.low_slope, res.high_slope

        if slope < -0.02:
            verdict = "IMPROVING"
        elif slope > 0.02:
            verdict = "WORSENING"
        else:
            verdict = "STAGNANT"

        return {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "lower_ci": round(float(low_slope), 4),
            "upper_ci": round(float(high_slope), 4),
            "verdict": verdict,
            "learning_velocity": round(float(-slope), 4)  # Positive velocity means error reduction
        }

    @staticmethod
    def bkt_update(
        prior_mastery: float,
        is_correct: bool,
        p_t: float = 0.15,
        p_g: float = 0.10,
        p_s: float = 0.05
    ) -> Dict[str, Any]:
        """
        Standard 4-parameter Bayesian Knowledge Tracing posterior update.
        
        Args:
            prior_mastery: P(L_{t-1})
            is_correct: Observed performance on the specific Akshara/rule
            p_t: Transition probability (learning rate)
            p_g: Guess probability
            p_s: Slip probability
        """
        p_l = max(0.01, min(0.99, prior_mastery))
        
        if is_correct:
            # P(L_t | Correct)
            num = p_l * (1.0 - p_s)
            denom = num + (1.0 - p_l) * p_g
            p_obs = num / max(1e-6, denom)
        else:
            # P(L_t | Incorrect)
            num = p_l * p_s
            denom = num + (1.0 - p_l) * (1.0 - p_g)
            p_obs = num / max(1e-6, denom)
            
        # Skill transition for next trial
        posterior = p_obs + (1.0 - p_obs) * p_t
        posterior = round(float(min(0.999, max(0.001, posterior))), 4)
        
        mastered = bool(posterior >= 0.85)

        return {
            "prior": prior_mastery,
            "observation": "CORRECT" if is_correct else "INCORRECT",
            "posterior_mastery": posterior,
            "mastered": mastered,
            "mastery_threshold": 0.85
        }

    @staticmethod
    def cusum_change_point(
        error_rates: List[float],
        target_mean: float = 0.30,
        k: float = 0.05,
        h: float = 0.20
    ) -> Dict[str, Any]:
        """
        Two-sided Cumulative Sum Control Chart to detect sudden shifts.
        S+: Detects upward surge (acute regression)
        S-: Detects downward drop (breakthrough milestone)
        """
        s_pos = 0.0
        s_neg = 0.0
        alerts = []

        for idx, val in enumerate(error_rates):
            # Regression check (errors increasing)
            s_pos = max(0.0, s_pos + (val - target_mean - k))
            # Breakthrough check (errors decreasing)
            s_neg = max(0.0, s_neg - (val - target_mean + k))

            if s_pos > h:
                alerts.append({
                    "session_index": idx + 1,
                    "type": "REGRESSION_SURGE",
                    "s_score": round(float(s_pos), 3)
                })
                s_pos = 0.0  # Reset upon alarm

            if s_neg > h:
                alerts.append({
                    "session_index": idx + 1,
                    "type": "BREAKTHROUGH_DROP",
                    "s_score": round(float(s_neg), 3)
                })
                s_neg = 0.0  # Reset upon alarm

        primary_alert = "NONE"
        if alerts:
            primary_alert = alerts[-1]["type"]

        return {
            "current_s_pos": round(float(s_pos), 3),
            "current_s_neg": round(float(s_neg), 3),
            "threshold_h": h,
            "alert": primary_alert,
            "history": alerts
        }
