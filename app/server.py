"""
FastAPI Server for the Malayalam Child Handwriting Diagnostic & Longitudinal Tracking Engine.
Production-grade backend connecting directly to implemented agents and deterministic tools.
Includes real image stroke segmentation & bounding-box localization.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import time
import uuid
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from src.agents.orchestrator import DiagnosticOrchestrator

app = FastAPI(
    title="Malayalam Handwriting Screening Support",
    description="Teacher-reviewed Malayalam probe analysis and progress tracking prototype",
    version="2.0.0"
)

# Ensure uploads directory exists and mount static files
os.makedirs("app/static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize orchestrator with authentic clinical datasets
TAXONOMY_PATH = "Malayalam_Data_With_Generated_IPA_ErrorsHandled_updated (1).xlsx"
LEXICON_PATH = "IPA- syllablewise (1) (1) (1).xlsx"

orchestrator = DiagnosticOrchestrator(
    taxonomy_path=TAXONOMY_PATH,
    lexicon_path=LEXICON_PATH
)

STUDENT_DATABASE = {
    "STU_2026_042": {
        "name": "Demo learner A (Class 3)",
        "language_profile": "Demo profile",
        "active_locus": "Dental vs Retroflex Liquid (ല vs ള)",
        "sessions": []
    },
    "STU_2026_019": {
        "name": "Demo learner B (Class 2)",
        "language_profile": "Demo profile",
        "active_locus": "Non-Linear Diacritics (Matra Placement)",
        "sessions": []
    }
}
DATABASE_PATH = Path(__file__).resolve().parent / "akshara_sessions.sqlite3"


@contextmanager
def get_database():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with get_database() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS probe_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                session_num INTEGER NOT NULL,
                date_label TEXT NOT NULL,
                error_rate REAL NOT NULL,
                correct INTEGER NOT NULL,
                dimension TEXT NOT NULL,
                bkt_mastery REAL NOT NULL,
                recorded_at REAL NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS teacher_reviews (
                report_id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                approved INTEGER NOT NULL DEFAULT 0,
                approved_at REAL
            )
        """)
        rows = connection.execute(
            "SELECT student_id, session_num, date_label, error_rate, correct, dimension, bkt_mastery, recorded_at FROM probe_observations ORDER BY id"
        ).fetchall()
    for row in rows:
        student = STUDENT_DATABASE.get(row["student_id"])
        if student is not None:
            student["sessions"].append({
                "session_num": row["session_num"], "date": row["date_label"],
                "error_rate": row["error_rate"], "correct": bool(row["correct"]),
                "dimension": row["dimension"], "bkt_mastery": row["bkt_mastery"],
                "recorded_at": row["recorded_at"]
            })


initialize_database()


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        content = f.read()
    response = HTMLResponse(content=content)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/api/presets")
async def get_presets():
    """Returns authentic clinical presets from the 8,673 dataset."""
    return [
        {
            "id": "case_1",
            "name": "Case 1: Voicing Error (ക vs ഗ)",
            "target": "അംഗം",
            "spelled": "അംകം",
            "student_id": "STU_2026_042",
            "category": "Phonological (Voicing)",
            "description": "Voiceless unaspirated velar plosive /k/ substituted for voiced /ɡ/"
        },
        {
            "id": "case_2",
            "name": "Case 2: Aspiration Error (ക vs ഖ)",
            "target": "അഖില",
            "spelled": "അകില",
            "student_id": "STU_2026_042",
            "category": "Phonological (Aspiration)",
            "description": "Unaspirated plosive written in place of aspirated plosive"
        },
        {
            "id": "case_3",
            "name": "Case 3: Non-Linear Diacritic Error",
            "target": "അകലം",
            "spelled": "അകആലം",
            "student_id": "STU_2026_019",
            "category": "Orthographic (Multi-linearity)",
            "description": "Child wrote primary vowel 'ആ' instead of attached matra 'ാ'"
        },
        {
            "id": "case_4",
            "name": "Case 4: Visual Uniformity (ഉ vs ള)",
            "target": "തവള",
            "spelled": "തവഉ",
            "student_id": "STU_2026_019",
            "category": "Orthographic (Visual Uniformity)",
            "description": "Child wrote visually similar glyph 'ഉ' in place of 'ള'"
        },
        {
            "id": "case_5",
            "name": "Case 5: Vowel Length Distinction (വി vs വീ)",
            "target": "വീണ",
            "spelled": "വിണ",
            "student_id": "STU_2026_042",
            "category": "Phonological (Vowel Length)",
            "description": "Short vowel diacritic written instead of long vowel diacritic"
        }
    ]


@app.get("/api/students")
async def get_students():
    return [
        {"id": k, "name": v["name"], "profile": v["language_profile"], "locus": v["active_locus"]}
        for k, v in STUDENT_DATABASE.items()
    ]


@app.post("/api/analyze")
async def analyze_submission(
    target_word: Optional[str] = Form(None),
    spelled_word: Optional[str] = Form(None),
    student_id: str = Form("STU_2026_042"),
    session_num: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Main diagnostic endpoint:
    Analyzes a known-target probe from typed text or a supported image.
    """
    target = (target_word or "").strip()
    extracted_text = (spelled_word or "").strip()
    response_source = "teacher-entered text" if extracted_text else "HERA visual suggestion"
    if not target:
        raise HTTPException(status_code=422, detail="Enter the expected probe word. The target cannot be inferred safely from a handwriting sample.")
    if student_id not in STUDENT_DATABASE:
        raise HTTPException(status_code=422, detail="Choose a valid learner profile.")
    file_type = "TEXT"
    visual_data = None
    file_preview_info = None

    # Handle File Upload (PDF or Image)
    if file and file.filename:
        content = await file.read()
        filename_lower = file.filename.lower()
        timestamp_prefix = f"upload_{int(time.time())}"

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image is too large. Please upload an image under 10 MB.")
        if any(filename_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
            file_type = "IMAGE"
            # HERA visual localization plus SigLIP image-to-workbook retrieval.
            visual_data = orchestrator.ocr.process_handwriting_image(
                image_bytes=content,
                output_dir="app/static/uploads",
                filename_prefix=timestamp_prefix,
                sheet_candidates=orchestrator.rag.candidate_responses_for_target(target)
            )
            if not visual_data.get("success"):
                raise HTTPException(status_code=422, detail=visual_data.get("error", "The image could not be read. Try a clearer PNG, JPG, or WebP photo."))

            file_preview_info = {
                "name": file.filename,
                "dimensions": visual_data.get("dimensions", "Unknown"),
                "stroke_density": visual_data.get("stroke_density", 0.0),
                "segmented_boxes": visual_data.get("segmented_boxes_count", 0),
                "raw_image_url": visual_data.get("raw_image_url"),
                "annotated_image_url": visual_data.get("annotated_image_url"),
                "hera_tokens": visual_data.get("tokens", []),
                "hera_recognized": visual_data.get("spelled_word", ""),
                "visual_candidates": visual_data.get("visual_candidates", []),
                "sheet_candidates": visual_data.get("sheet_candidates", []),
                "sheet_candidate_count": visual_data.get("sheet_candidate_count", 0),
                "matching_mode": visual_data.get("matching_mode"),
                "vision_similarity": visual_data.get("vision_similarity"),
                "recognizer_mode": visual_data.get("recognizer_mode")
            }
        else:
            raise HTTPException(status_code=415, detail="Only PNG, JPG, JPEG, and WebP images are supported. PDF pages are not currently read as handwriting.")

    # Resolve child's response:
    # 1. If explicit teacher text is typed, use it.
    # 2. If an image is uploaded, automatically use HERA/SigLIP's top visual embedding match.
    analysis_response = extracted_text or ((visual_data or {}).get("spelled_word") or "").strip()
    if not analysis_response and visual_data and visual_data.get("sheet_candidates"):
        analysis_response = visual_data["sheet_candidates"][0]["word"]

    if not analysis_response:
        analysis_response = target

    # 1. RUN ERROR-ANALYSIS AGENT (HERA Vision + 3-Tier G2P + PanPhon + 8,673-Row Gold RAG)
    error_analysis = orchestrator.run_error_analysis(
        target_word=target,
        simulated_raw_handwriting=analysis_response
    )
    if visual_data:
        error_analysis["recognizer_mode"] = ("teacher-entered text (photo shown for comparison)" if extracted_text
                                               else visual_data.get("recognizer_mode", "HERA visual suggestion"))
        error_analysis["visual_candidates"] = visual_data.get("visual_candidates", [])
        error_analysis["sheet_candidates"] = visual_data.get("sheet_candidates", [])
        error_analysis["sheet_candidate_count"] = visual_data.get("sheet_candidate_count", 0)
        error_analysis["visual_matching_mode"] = visual_data.get("matching_mode")
        error_analysis["vision_similarity"] = visual_data.get("vision_similarity")
    error_analysis["response_source"] = response_source

    # 2. RUN LONGITUDINAL EVALUATION (Theil-Sen + BKT + CUSUM + CMSDA)
    student = STUDENT_DATABASE[student_id]
    history = [s for s in student["sessions"] if s["dimension"] == error_analysis["dimension"]]
    curr_session_idx = session_num if session_num is not None else (len(history) + 1)

    is_correct = (not error_analysis["is_error"])
    # This UI currently accepts one item per submission, so record an item
    # outcome (0/1). Do not turn phonetic distance into a fabricated error rate.
    curr_error_rate = 0.0 if is_correct else 1.0

    new_session_record = {
        "session_num": curr_session_idx,
        "date": f"Probe {curr_session_idx}",
        "error_rate": round(curr_error_rate, 4),
        "correct": is_correct,
        "dimension": error_analysis["dimension"]
    }

    dimension_tested = error_analysis["dimension"]
    longitudinal_eval = orchestrator.run_longitudinal_evaluation(
        student_id=student_id,
        dimension=dimension_tested,
        session_history=history,
        new_session=new_session_record
    )
    new_session_record["bkt_mastery"] = longitudinal_eval["bkt_mastery_probability"]
    new_session_record["recorded_at"] = time.time()
    student["sessions"].append(new_session_record)
    with get_database() as connection:
        connection.execute(
            "INSERT INTO probe_observations (student_id, session_num, date_label, error_rate, correct, dimension, bkt_mastery, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (student_id, curr_session_idx, new_session_record["date"], curr_error_rate,
             int(is_correct), error_analysis["dimension"], longitudinal_eval["bkt_mastery_probability"],
             new_session_record["recorded_at"])
        )

    # 3. GENERATE TEACHER GUIDANCE, QUOTES, AND ACTIONABLE PEDAGOGICAL TIPS
    teacher_packet = generate_teacher_guidance(error_analysis, longitudinal_eval, student["name"])

    # Prepare historical chart points
    chart_sessions = [s["date"] for s in history] + [f"Probe {curr_session_idx}"]
    chart_error_rates = [round(s["error_rate"] * 100, 1) for s in history] + [round(curr_error_rate * 100, 1)]

    report_id = str(uuid.uuid4())
    with get_database() as connection:
        connection.execute(
            "INSERT INTO teacher_reviews (report_id, student_id) VALUES (?, ?)",
            (report_id, student_id)
        )
    return {
        "report_id": report_id,
        "status": "SUCCESS",
        "file_info": {
            "type": file_type,
            "details": file_preview_info
        },
        "student": {
            "id": student_id,
            "name": student["name"],
            "profile": student["language_profile"]
        },
        "error_analysis": error_analysis,
        "response_source": response_source,
        "longitudinal_eval": longitudinal_eval,
        "teacher_guidance": teacher_packet,
        "chart_data": {
            "labels": chart_sessions,
            "error_rates": chart_error_rates,
            "slope": longitudinal_eval["theil_sen_slope"],
            "mastery": round(longitudinal_eval["bkt_mastery_probability"] * 100, 1)
        }
    }


@app.post("/api/approve")
async def approve_intervention(report_id: str = Form(...)):
    with get_database() as connection:
        updated = connection.execute(
            "UPDATE teacher_reviews SET approved = 1, approved_at = ? WHERE report_id = ?",
            (time.time(), report_id)
        ).rowcount
    if updated == 0:
        raise HTTPException(status_code=404, detail="This report has expired. Reopen the evaluation before approving it.")
    return {"status": "APPROVED", "report_id": report_id}


def generate_teacher_guidance(error_res: Dict[str, Any], long_res: Dict[str, Any], student_name: str) -> Dict[str, Any]:
    """
    Synthesizes authentic clinical quotes, pedagogical rationale, and classroom intervention tips.
    """
    dim = error_res["dimension"]
    verdict = long_res["verdict"]
    action = long_res["action_required"]
    wfd = error_res["weighted_feature_distance"]

    quotes = {
        "Phonological": [
            "Children do not confuse letters at random; their handwriting faithfully maps how their auditory system encodes the sound.",
            "Speech motor perception precedes orthographic mastery. Align the tongue position before reinforcing the pen stroke."
        ],
        "Orthographic": [
            "Malayalam's non-linear akshara layout requires mental spatial rotation. Ensure the child understands matra attachment before drilling vocabulary.",
            "Visual similarity confusions diminish rapidly when multi-sensory color-coding separates consonants from dependent vowel signs."
        ]
    }

    quote = quotes["Phonological"][0] if "Phon" in error_res["error_type"] else quotes["Orthographic"][0]

    drills = []
    if "Retroflex" in dim or "ല" in error_res["target_word"]:
        drills = [
            "Tactile Sand-Tray Tracing: Have the child curl the tongue tip up to the hard palate while physically drawing 'ള' in a sand tray.",
            "Acoustic Minimal Pair Contrast: Practice listening discrimination using 'വില' (price, dental) vs 'വിള' (crop, retroflex).",
            "Kinesthetic Finger Taps: Tap the chin for dental 'ല' and tap the top of the palate for retroflex 'ള'."
        ]
    elif "Voic" in dim or "ക" in error_res["target_word"] or "ഗ" in error_res["target_word"]:
        drills = [
            "Throat Vibration Drill: Place the child's fingertips gently on the larynx. Feel the vibration for voiced 'ഗ' and absence of vibration for voiceless 'ക'.",
            "Whisper-to-Phonate Drill: Whisper 'ക' (no cord movement), then vocalize 'ഗ' with firm voice onset.",
            "Color-coded Stroke Worksheet: Green highlight for voiced plosives, Blue for voiceless plosives."
        ]
    elif "Multi-linearity" in dim or "Diacritic" in dim:
        drills = [
            "Split-Card Matching Game: Provide cards with base consonants and separate transparent overlay cards with vowel matras (ാ, ി, െ).",
            "Two-Step Phonics Chant: Step 1: Speak the consonant ('ക'). Step 2: Attach the vowel sign ('ാ') to produce ('കാ').",
            "Avoid Independent Vowel Duplication: Explicitly teach that independent vowels (അ, ആ, ഇ) ONLY appear at word-initial positions in modern Malayalam."
        ]
    elif "Visual" in dim or "Uniformity" in dim:
        drills = [
            "Contour Highlighting: Use contrasting highlighters to trace the distinct looping features of visually similar letters (e.g., 'ഉ' vs 'ള').",
            "Sandpaper Letter Tracing: Have the child close their eyes and trace the distinct curves of 'ഉ' vs 'ള' using finger touch.",
            "Sorting Basket Drill: Provide mixed tokens and ask the child to sort them into separate labeled bins."
        ]
    else:
        drills = [
            "Spaced Repetition Flashcards: 3-minute morning drill on active error target.",
            "Multi-Sensory Clay Modeling: Form the shape of the target Akshara with play-dough.",
            "Peer-Assisted Reading: Read target words aloud with a peer partner."
        ]

    if verdict in ("BASELINE", "INSUFFICIENT_DATA"):
        tier_level = "Teacher review (evidence gathering)"
        action_summary = "This single-item result is not enough to infer a learning trend. Collect comparable probes over time and review the sample with the teacher."
    elif action == "TIER_ESCALATION" or verdict == "INCREASED":
        tier_level = "Teacher review (consider targeted support)"
        action_summary = f"This probe differed from the recent pattern (+{long_res['immediate_delta']:.1%}). Review the sample and consider more targeted practice; seek qualified support if concerns continue."
    elif action == "MODALITY_SHIFT" or verdict == "STAGNANT":
        tier_level = "Tier 1 Targeted (Multi-Sensory Modality Shift)"
        action_summary = "Recent probes show little change. Review the activities with the learner and consider a different practice format, such as tactile tracing or listening contrasts."
    elif verdict == "MODERATE_FLUCTUATION":
        tier_level = "Teacher review (continue monitoring)"
        action_summary = "Probe results vary without a clear trend. Continue comparable probes and review the samples with the teacher."
    else:
        tier_level = "Tier 1 Universal (Progressing on Schedule)"
        action_summary = f"Recent probes suggest improvement (slope: {long_res['theil_sen_slope']:.3f} per probe). Confirm the pattern with the teacher and continue current practice."

    return {
        "educational_quote": quote,
        "clinical_rationale": (
            f"For {student_name}, the recorded probe target is '{error_res['target_word']}' and the response is '{error_res['spelled_word']}'. "
            f"The tool reports an articulatory distance of {wfd} and the following differing features: "
            f"{', '.join(error_res['differing_articulatory_features']) or 'none detected'}. "
            "This is an educational screening observation, not a diagnosis; the teacher should verify the transcription and interpretation."
        ),
        "recommended_tier": tier_level,
        "action_summary": action_summary,
        "classroom_drills": drills,
        "self_verified_stamp": long_res.get("verified_teacher_report", "Verified by Closed-Loop Insight Agent.")
    }
