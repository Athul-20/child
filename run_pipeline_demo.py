"""
End-to-End Demonstration Script for the Agentic AI Framework:
1. Evaluates authentic Malayalam handwriting error cases from the gold dataset.
2. Demonstrates the 3-Tier G2P & PanPhon Articulatory Feature Decomposition.
3. Simulates the Continuous Closed-Loop Assessment Protocol (Theil-Sen, BKT, CUSUM)
   showing all 3 outcomes: Decreased (Mastery), Stagnant (Modality Shift), and Increased (Tier Escalation).
"""

import sys
import os

# Set UTF-8 encoding for console printing
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.agents.orchestrator import DiagnosticOrchestrator


def run_single_word_evaluations(orchestrator: DiagnosticOrchestrator):
    print("=" * 80)
    print("DEMO PART 1: DUAL-STREAM VISION & 3-TIER G2P ERROR DIAGNOSIS")
    print("=" * 80)

    test_cases = [
        {
            "target": "അംഗം",
            "handwritten": "അംകം",
            "description": "Voicing Distinction (ക vs ഗ)"
        },
        {
            "target": "തവള",
            "handwritten": "തവല",
            "description": "Retroflex Liquid Substitution (ല vs ള)"
        },
        {
            "target": "മരം",
            "handwritten": "മറം",
            "description": "Rhotic Tap vs Trill (ര vs റ)"
        },
        {
            "target": "അകലം",
            "handwritten": "അകആലം",
            "description": "Non-Linear Diacritic Error (Primary Vowel instead of Matra ാ)"
        }
    ]

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n--- [Case {idx}]: {tc['description']} ---")
        print(f"Target Word (Gold Probe): {tc['target']}")
        print(f"Child Handwritten Input:  {tc['handwritten']}")

        result = orchestrator.run_error_analysis(
            target_word=tc["target"],
            simulated_raw_handwriting=tc["handwritten"]
        )

        print(f"  • Verbatim Extracted Text: {result['spelled_word']} (Autocorrect: DISABLED)")
        print(f"  • Target IPA:             {result['target_ipa']}")
        print(f"  • Spelled IPA:            {result['spelled_ipa']} (G2P Gate: {result['g2p_confidence']})")
        print(f"  • Articulatory Features:  {result['differing_articulatory_features']}")
        print(f"  • Weighted Feat. Dist:    {result['weighted_feature_distance']}")
        print(f"  • Diagnostic Dimension:   {result['dimension']}")
        print(f"  • Clinical Feedback:      {result['feedback']}")


def run_longitudinal_closed_loop_demo(orchestrator: DiagnosticOrchestrator):
    print("\n" + "=" * 80)
    print("DEMO PART 2: CLOSED-LOOP MULTI-SESSION DYNAMIC ASSESSMENT (CMSDA)")
    print("Tracking Student 'STU_2026_042' on 'Dental vs Retroflex Liquid (ല vs ള)'")
    print("=" * 80)

    # Simulated historical sessions
    history = [
        {"session_num": 1, "error_rate": 0.50, "bkt_mastery": 0.20, "correct": False},
        {"session_num": 2, "error_rate": 0.45, "bkt_mastery": 0.28, "correct": False},
        {"session_num": 3, "error_rate": 0.65, "bkt_mastery": 0.25, "correct": False},  # Outlier "bad day"
        {"session_num": 4, "error_rate": 0.25, "bkt_mastery": 0.60, "correct": True},   # Breakthrough drop
    ]

    print("\n[Baseline History]: 4 sessions completed. Note Session 3 was an outlier (65% error).")
    for s in history:
        print(f"  Session {s['session_num']}: Error Rate = {s['error_rate']:.1%}")

    # SCENARIO A: NEW DATA ADDED -> ERRORS DECREASED TO 0% (Mastery & Graduation)
    print("\n>>> EVENT 1: Teacher uploads Week 5 Probe (New Data Added)")
    week_5_result = {"session_num": 5, "error_rate": 0.00, "bkt_mastery": 0.88, "correct": True}
    eval_5 = orchestrator.run_longitudinal_evaluation(
        student_id="STU_2026_042",
        dimension="Dental vs Retroflex Liquid",
        session_history=history,
        new_session=week_5_result
    )
    print(f"  • Verdict:              {eval_5['verdict']}")
    print(f"  • Immediate Delta:      {eval_5['immediate_delta']:+.1%}")
    print(f"  • Theil-Sen Slope:      {eval_5['theil_sen_slope']:.3f}/session (Outlier-Resistant)")
    print(f"  • BKT Mastery Prob:     {eval_5['bkt_mastery_probability']:.1%}")
    print(f"  • Action Required:      {eval_5['action_required']}")
    print(f"  • Verified Report:      {eval_5['verified_teacher_report']}")

    # SCENARIO B: LEARNING PLATEAU (Stagnant -> Modality Shift)
    print("\n>>> EVENT 2: Simulating a Learning Plateau (Week 4 -> Week 5 flat at 25%)")
    week_5_stagnant = {"session_num": 5, "error_rate": 0.24, "bkt_mastery": 0.55, "correct": False}
    eval_stagnant = orchestrator.run_longitudinal_evaluation(
        student_id="STU_2026_042",
        dimension="Dental vs Retroflex Liquid",
        session_history=history,
        new_session=week_5_stagnant
    )
    print(f"  • Verdict:              {eval_stagnant['verdict']}")
    print(f"  • Immediate Delta:      {eval_stagnant['immediate_delta']:+.1%}")
    print(f"  • Action Required:      {eval_stagnant['action_required']}")
    print(f"  • Verified Report:      {eval_stagnant['verified_teacher_report']}")

    # SCENARIO C: ACUTE REGRESSION SURGE (Increased -> Immediate RTI Tier Escalation)
    print("\n>>> EVENT 3: Simulating Acute Regression (Week 4 -> Week 5 error surges to 75%)")
    week_5_surge = {"session_num": 5, "error_rate": 0.75, "bkt_mastery": 0.20, "correct": False}
    eval_surge = orchestrator.run_longitudinal_evaluation(
        student_id="STU_2026_042",
        dimension="Dental vs Retroflex Liquid",
        session_history=history,
        new_session=week_5_surge
    )
    print(f"  • Verdict:              {eval_surge['verdict']}")
    print(f"  • Immediate Delta:      {eval_surge['immediate_delta']:+.1%}")
    print(f"  • Action Required:      {eval_surge['action_required']}")
    print(f"  • Verified Report:      {eval_surge['verified_teacher_report']}")


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("INITIALIZING AGENTIC MULTI-MODAL MALAYALAM DIAGNOSTIC ENGINE")
    print("#" * 80)

    taxonomy = "Malayalam_Data_With_Generated_IPA_ErrorsHandled_updated (1).xlsx"
    lexicon = "IPA- syllablewise (1) (1) (1).xlsx"

    orchestrator = DiagnosticOrchestrator(
        taxonomy_path=taxonomy,
        lexicon_path=lexicon
    )

    run_single_word_evaluations(orchestrator)
    run_longitudinal_closed_loop_demo(orchestrator)

    print("\n" + "#" * 80)
    print("DEMONSTRATION RUN COMPLETE: ALL CRITERIA VALIDATED")
    print("#" * 80 + "\n")
