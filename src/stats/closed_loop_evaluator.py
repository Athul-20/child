"""
Closed-Loop Multi-Session Dynamic Assessment (CMSDA) Protocol Engine.
Implements the continuous evaluation feedback loop:
- Triggered whenever new assessment data is added
- Compares new data against historical baseline
- Evaluates whether errors have:
    1. Decreased (Improving) -> Graduate concept & advance difficulty
    2. Stayed the Same (Stagnant) -> Trigger Modality Shift
    3. Increased (Worsening) -> Escalate RTI Tier
"""

from typing import Dict, Any, List
import numpy as np
from src.stats.longitudinal_engine import LongitudinalStatsEngine


class ClosedLoopEvaluator:
    """
    Closed-Loop Action Engine driving automated teacher recommendations.
    """

    def __init__(self):
        self.stats = LongitudinalStatsEngine()

    def evaluate_session_delta(
        self,
        student_id: str,
        dimension: str,
        historical_sessions: List[Dict[str, Any]],
        current_session_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes the closed-loop evaluation cycle upon receiving new handwriting data.
        
        Args:
            student_id: Identifier of the child
            dimension: Specific linguistic error category (e.g. "Dental vs Retroflex Liquid")
            historical_sessions: List of past session records [{"session_num": int, "error_rate": float, "correct": bool}]
            current_session_result: Dict for the newest session {"session_num": int, "error_rate": float, "correct": bool}
        """
        all_sessions = historical_sessions + [current_session_result]
        session_indices = [s["session_num"] for s in all_sessions]
        error_rates = [s["error_rate"] for s in all_sessions]

        if len(historical_sessions) < 1:
            initial_bkt = self.stats.bkt_update(
                prior_mastery=0.20,
                is_correct=current_session_result.get("correct", current_session_result["error_rate"] == 0.0)
            )
            return {
                "student_id": student_id,
                "dimension": dimension,
                "status": "BASELINE_INITIALIZED",
                "immediate_delta": 0.0,
                "theil_sen_slope": 0.0,
                "learning_velocity": 0.0,
                "verdict": "BASELINE",
                "action_required": "CONTINUE_BASELINE",
                "bkt_mastery_probability": initial_bkt["posterior_mastery"],
                "cusum_alert": "NONE",
                "teacher_recommendation": "Baseline assessment recorded. Future probe submissions will track longitudinal slope and concept mastery.",
                "verified_teacher_report": "Baseline assessment recorded. Future probe submissions will track longitudinal slope and concept mastery.",
                "sessions_completed": 1
            }

        # 1. Immediate Delta
        current_err = current_session_result["error_rate"]
        previous_err = historical_sessions[-1]["error_rate"]
        immediate_delta = round(current_err - previous_err, 4)

        # 2. Theil-Sen Robust Learning Velocity
        theil_res = self.stats.theil_sen_slope(session_indices, error_rates)
        slope = theil_res["slope"]

        # 3. Bayesian Knowledge Tracing (BKT) Update
        prior_bkt = historical_sessions[-1].get("bkt_mastery", 0.20)
        bkt_res = self.stats.bkt_update(
            prior_mastery=prior_bkt,
            is_correct=current_session_result.get("correct", (current_err == 0.0))
        )
        posterior = bkt_res["posterior_mastery"]

        # 4. CUSUM Sequential Test
        cusum_res = self.stats.cusum_change_point(
            error_rates=error_rates,
            target_mean=float(np.mean(error_rates[:-1])) if len(error_rates) > 1 else current_err
        )

        # ─── CASE A: DECREASED (Improving) ──────────────────────────────────
        if (immediate_delta < -0.08 and slope < -0.01) or cusum_res["alert"] == "BREAKTHROUGH_DROP":
            verdict = "DECREASED"
            if posterior >= 0.85:
                action = "GRADUATE_CONCEPT"
                recommendation = (
                    f"Mastery confirmed ({posterior:.1%}). The child has mastered the {dimension} rule. "
                    "Graduate this target and advance to higher-complexity conjunct Aksharas (കൂട്ടക്ഷരങ്ങൾ)."
                )
            else:
                action = "MAINTAIN_INTERVENTION"
                recommendation = (
                    f"Statistically significant error reduction (Rate: {slope:+.3f}/session). "
                    "The current pedagogical intervention is working effectively. Continue current practice protocol."
                )

        # ─── CASE B: STAGNANT (Learning Plateau) ────────────────────────────
        elif abs(immediate_delta) <= 0.06 and abs(slope) <= 0.02:
            verdict = "STAGNANT"
            action = "MODALITY_SHIFT"
            recommendation = (
                f"Learning plateau detected across {len(all_sessions)} sessions (Error rate flat at {current_err:.1%}). "
                "Shift pedagogical modality: Transition from standard 2D visual worksheets to 3D multi-sensory "
                "tactile clay-modelling and kinesthetic sand-tray tracing for this Akshara."
            )

        # ─── CASE C: INCREASED (Worsening / Acute Regression) ───────────────
        elif immediate_delta > 0.08 or slope > 0.02 or cusum_res["alert"] == "REGRESSION_SURGE":
            verdict = "INCREASED"
            action = "TIER_ESCALATION"
            recommendation = (
                f"ALERT: Error rate increased by {immediate_delta:+.1%} (CUSUM regression spike detected). "
                "Escalate intervention tier from Tier 1 classroom practice to Tier 2 small-group intensive "
                "phonological remediation. Schedule an individual teacher diagnostic review."
            )

        else:
            verdict = "MODERATE_FLUCTUATION"
            action = "MONITOR"
            recommendation = "Minor variance observed within expected tolerance limits. Maintain current drill intensity."

        return {
            "student_id": student_id,
            "dimension": dimension,
            "sessions_completed": len(all_sessions),
            "verdict": verdict,
            "immediate_delta": immediate_delta,
            "theil_sen_slope": slope,
            "learning_velocity": theil_res["learning_velocity"],
            "bkt_mastery_probability": posterior,
            "cusum_alert": cusum_res["alert"],
            "action_required": action,
            "teacher_recommendation": recommendation,
            "verified_teacher_report": recommendation
        }

    # Alias for method compatibility
    evaluate_session_transition = evaluate_session_delta


import numpy as np
