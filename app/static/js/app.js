/**
 * Akshara — User-Friendly Frontend Controller
 * Manages run history, easy teacher explanations, quick examples, and file uploads.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const form = document.getElementById("evalForm");
    const submitBtn = document.getElementById("submitBtn");
    const loadingSpinner = document.getElementById("loadingSpinner");
    const targetWordInput = document.getElementById("targetWord");
    const spelledWordInput = document.getElementById("spelledWord");
    const imageSuggestionReview = document.getElementById("imageSuggestionReview");
    const studentSelect = document.getElementById("studentSelect");
    const studentFocusBadge = document.getElementById("studentFocusBadge");
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const dropzonePreview = document.getElementById("dropzonePreview");
    const thumbImg = document.getElementById("thumbImg");
    const fileNameSpan = document.getElementById("fileName");
    const removeFileBtn = document.getElementById("removeFileBtn");
    const newEvalBtn = document.getElementById("newEvalBtn");
    const quickExamplesList = document.getElementById("quickExamplesList");
    const historyList = document.getElementById("historyList");
    const historyCount = document.getElementById("historyCount");
    const resultsSection = document.getElementById("resultsSection");
    const approveInterventionBtn = document.getElementById("approveInterventionBtn");
    const approvalStatus = document.getElementById("approvalStatus");
    const pipelineNotice = document.getElementById("pipelineNotice");

    let progressChart = null;
    let evalHistory = [];
    let currentReportId = null;

    // Load history from localStorage
    loadHistory();

    // ══════════════════════════════════════════════════════════════════════════
    // 1. STUDENT PROFILE SWITCHER
    // ══════════════════════════════════════════════════════════════════════════
    if (studentSelect) {
        studentSelect.addEventListener("change", () => {
            if (studentFocusBadge) {
                if (studentSelect.value === "STU_2026_042") {
                    studentFocusBadge.textContent = "Focus: Voicing (ക/ഗ) & Liquid (ല/ള)";
                } else {
                    studentFocusBadge.textContent = "Focus: Non-linear Matras (Multi-linearity)";
                }
            }
        });
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2. QUICK EXAMPLE SELECTION
    // ══════════════════════════════════════════════════════════════════════════
    if (quickExamplesList) {
        quickExamplesList.addEventListener("click", (e) => {
            const item = e.target.closest(".example-item");
            if (!item) return;

            document.querySelectorAll(".example-item").forEach(el => el.classList.remove("active"));
            item.classList.add("active");

            if (targetWordInput) targetWordInput.value = item.dataset.target || "";
            if (spelledWordInput) spelledWordInput.value = item.dataset.spelled || "";
            if (item.dataset.student && studentSelect) {
                studentSelect.value = item.dataset.student;
                studentSelect.dispatchEvent(new Event("change"));
            }
            clearSuggestionReview();
            clearFileInput();
        });
    }

    const resetHistoryBtn = document.getElementById("resetHistoryBtn");
    if (resetHistoryBtn) {
        resetHistoryBtn.addEventListener("click", async () => {
            if (!confirm("Reset student longitudinal histories back to canonical 4-week baselines?")) return;
            try {
                const res = await fetch("/api/students/reset", { method: "POST" });
                const d = await res.json();
                alert(d.message || "Baselines reset successfully!");
                location.reload();
            } catch (err) {
                alert("Failed to reset baselines: " + err.message);
            }
        });
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 3. FILE DRAG & DROP
    // ══════════════════════════════════════════════════════════════════════════
    if (dropZone) {
        dropZone.addEventListener("click", () => {
            if (fileInput) fileInput.click();
        });

        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "var(--primary)";
        });

        dropZone.addEventListener("dragleave", () => {
            dropZone.style.borderColor = "var(--border-light)";
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "var(--border-light)";
            if (e.dataTransfer && e.dataTransfer.files.length > 0) {
                handleSelectedFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                clearSuggestionReview();
                handleSelectedFile(e.target.files[0]);
            }
        });
    }

    if (removeFileBtn) {
        removeFileBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            clearFileInput();
        });
    }

    function handleSelectedFile(file) {
        // Drag-and-drop events do not populate the hidden file input by themselves.
        if (fileInput && (!fileInput.files.length || fileInput.files[0] !== file)) {
            const transfer = new DataTransfer();
            transfer.items.add(file);
            fileInput.files = transfer.files;
        }
        // A previous manually entered spelling must not silently override a
        // newly uploaded sample. Teachers can type an explicit correction
        // after choosing the photo if they want their transcription used.
        if (file.type.startsWith("image/") && spelledWordInput) spelledWordInput.value = "";
        if (fileNameSpan) fileNameSpan.textContent = file.name;
        document.querySelectorAll(".example-item").forEach(el => el.classList.remove("active"));

        if (file.type.startsWith("image/")) {
            const reader = new FileReader();
            reader.onload = (e) => {
                if (thumbImg) thumbImg.src = e.target.result;
                if (dropzonePreview) dropzonePreview.style.display = "flex";
                const dzBody = document.querySelector(".dropzone-body");
                if (dzBody) dzBody.style.display = "none";
            };
            reader.readAsDataURL(file);
        } else {
            if (thumbImg) thumbImg.src = "";
            if (dropzonePreview) dropzonePreview.style.display = "flex";
            const dzBody = document.querySelector(".dropzone-body");
            if (dzBody) dzBody.style.display = "none";
        }
    }

    function clearFileInput() {
        if (fileInput) fileInput.value = "";
        if (dropzonePreview) dropzonePreview.style.display = "none";
        const dzBody = document.querySelector(".dropzone-body");
        if (dzBody) dzBody.style.display = "flex";
        if (thumbImg) thumbImg.src = "";
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 4. NEW EVALUATION BUTTON
    // ══════════════════════════════════════════════════════════════════════════
    if (newEvalBtn) {
        newEvalBtn.addEventListener("click", () => {
            if (targetWordInput) targetWordInput.value = "";
            if (spelledWordInput) spelledWordInput.value = "";
            clearSuggestionReview();
            clearFileInput();
            if (resultsSection) resultsSection.style.display = "none";
            if (targetWordInput) targetWordInput.focus();
        });
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 5. FORM SUBMISSION
    // ══════════════════════════════════════════════════════════════════════════
    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();

            if (!targetWordInput || !targetWordInput.value.trim()) {
                alert("Enter the expected probe word before analyzing.");
                return;
            }
            const hasImage = Boolean(fileInput && fileInput.files.length > 0);
            if ((!spelledWordInput || !spelledWordInput.value.trim()) && !hasImage) {
                alert("Enter the child's written response or upload a handwriting photo.");
                return;
            }

            if (submitBtn) submitBtn.disabled = true;
            if (loadingSpinner) loadingSpinner.style.display = "inline-block";

            const formData = new FormData();
            formData.append("target_word", targetWordInput ? targetWordInput.value.trim() : "");
            formData.append("spelled_word", spelledWordInput ? spelledWordInput.value.trim() : "");
            formData.append("student_id", studentSelect ? studentSelect.value : "STU_2026_042");

            if (fileInput && fileInput.files.length > 0) {
                formData.append("file", fileInput.files[0]);
            }

            try {
                const response = await fetch("/api/analyze", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || "Could not process this probe.");

                // Auto-populate child's spelling with the HERA recognized text if input was left blank
                if (spelledWordInput && !spelledWordInput.value && data.error_analysis) {
                    spelledWordInput.value = data.error_analysis.spelled_word || "";
                }
                
                renderTeacherReport(data);
                saveToHistory(data);
            } catch (err) {
                alert("Error: " + err.message);
            } finally {
                if (submitBtn) submitBtn.disabled = false;
                if (loadingSpinner) loadingSpinner.style.display = "none";
            }
        });
    }

    if (approveInterventionBtn) {
        approveInterventionBtn.addEventListener("click", async () => {
            if (!currentReportId) return;
            approveInterventionBtn.disabled = true;
            try {
                const body = new FormData();
                body.append("report_id", currentReportId);
                const response = await fetch("/api/approve", { method: "POST", body });
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || "Could not record teacher review.");
                if (approvalStatus) approvalStatus.textContent = "Teacher approved the suggested practice for this probe.";
                approveInterventionBtn.textContent = "Approved";
            } catch (err) {
                approveInterventionBtn.disabled = false;
                alert(err.message);
            }
        });
    }

    function clearSuggestionReview() {
        if (!imageSuggestionReview) return;
        imageSuggestionReview.replaceChildren();
        imageSuggestionReview.style.display = "none";
    }

    function showImageSuggestionReview(candidates) {
        if (!imageSuggestionReview) return;
        imageSuggestionReview.replaceChildren();
        const intro = document.createElement("div");
        intro.textContent = "These are unverified SigLIP suggestions, not recognized text. Choose a candidate or type what the child wrote, then submit again to create the report.";
        imageSuggestionReview.appendChild(intro);
        const list = document.createElement("div");
        list.style.display = "flex";
        list.style.flexWrap = "wrap";
        list.style.gap = "8px";
        list.style.marginTop = "8px";
        candidates.forEach((candidate, index) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "optional-tag";
            button.style.cursor = "pointer";
            button.textContent = `${index + 1}. ${candidate.word}`;
            button.addEventListener("click", () => {
                if (spelledWordInput) {
                    spelledWordInput.value = candidate.word;
                    spelledWordInput.focus();
                }
            });
            list.appendChild(button);
        });
        imageSuggestionReview.appendChild(list);
        imageSuggestionReview.style.display = "block";
        imageSuggestionReview.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 6. RENDER TEACHER REPORT (USER-FRIENDLY)
    // ══════════════════════════════════════════════════════════════════════════
    function renderTeacherReport(data) {
        if (!resultsSection) return;
        currentReportId = data.report_id || null;
        if (approvalStatus) approvalStatus.textContent = "Review the evidence and suggested practice before using it.";
        if (approveInterventionBtn) {
            approveInterventionBtn.disabled = !currentReportId;
            approveInterventionBtn.textContent = "Approve suggested practice";
        }
        resultsSection.style.display = "flex";
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

        const err = data.error_analysis || {};
        const long = data.longitudinal_eval || {};
        const teacher = data.teacher_guidance || {};
        const chart = data.chart_data || {};
        if (pipelineNotice) {
            const imageInput = data.file_info && data.file_info.type === "IMAGE";
            pipelineNotice.textContent = imageInput
                ? "The uploaded photo was matched against workbook response candidates. The suggestions are not verified transcription; confirm the child's spelling before using the analysis."
                : "Prototype limitation: G2P uses a lexicon and rule fallback, not the OSUR neural checkpoint. Treat this as screening support and verify the interpretation with a teacher.";
        }

        // STATUS BANNER
        const banner = document.getElementById("statusBanner");
        const statusBadge = document.getElementById("statusBadge");
        const statusTitle = document.getElementById("statusTitle");
        const statusDesc = document.getElementById("statusDesc");
        const statusIcon = document.getElementById("statusIcon");

        if (banner && statusBadge && statusTitle && statusDesc && statusIcon) {
            if (long.verdict === "INSUFFICIENT_DATA" || long.verdict === "BASELINE") {
                banner.style.borderLeftColor = "var(--border-light)";
                statusBadge.className = "status-badge status-badge-stagnant";
                statusBadge.textContent = "Baseline";
                statusIcon.textContent = "i";
                statusTitle.textContent = "Baseline Probe Recorded";
                statusDesc.textContent = long.teacher_recommendation || "Baseline recorded. Subsequent probes will compute Theil-Sen learning velocity and BKT mastery.";
            } else if (long.verdict === "DECREASED") {
                banner.style.borderLeftColor = "var(--success)";
                statusBadge.className = "status-badge status-badge-improving";
                statusBadge.textContent = "Improving";
                statusIcon.textContent = "✓";
                statusIcon.style.color = "var(--success)";
                statusIcon.style.background = "var(--success-bg)";
                statusTitle.textContent = long.bkt_mastery_probability >= 0.85 ? "Great Progress! Concept Mastered" : "Steady Improvement Detected";
                statusDesc.textContent = long.teacher_recommendation || `The observed error rate fell by ${Math.abs(long.immediate_delta * 100).toFixed(0)}%. Continue current practice protocol.`;
            } else if (long.verdict === "STAGNANT") {
                banner.style.borderLeftColor = "var(--warning)";
                statusBadge.className = "status-badge status-badge-stagnant";
                statusBadge.textContent = "Plateau";
                statusIcon.textContent = "•";
                statusIcon.style.color = "var(--warning)";
                statusIcon.style.background = "var(--warning-bg)";
                statusTitle.textContent = "Learning Plateau (Modality Shift Required)";
                statusDesc.textContent = long.teacher_recommendation || "Error rate remained flat over consecutive sessions. Recommend multi-sensory sand tray tracing.";
            } else if (long.verdict === "INCREASED") {
                banner.style.borderLeftColor = "var(--danger)";
                statusBadge.className = "status-badge status-badge-regression";
                statusBadge.textContent = "Needs Attention";
                statusIcon.textContent = "!";
                statusIcon.style.color = "var(--danger)";
                statusIcon.style.background = "var(--danger-bg)";
                statusTitle.textContent = "Noticeable Error Surge (Tier Escalation)";
                statusDesc.textContent = long.teacher_recommendation || `The observed error rate rose by ${Math.abs(long.immediate_delta * 100).toFixed(0)}%. Escalate intervention tier from Tier 1 to Tier 2.`;
            } else {
                banner.style.borderLeftColor = "var(--border-light)";
                statusBadge.className = "status-badge status-badge-stagnant";
                statusBadge.textContent = "Monitoring";
                statusIcon.textContent = "•";
                statusTitle.textContent = "Fluctuating Performance";
                statusDesc.textContent = long.teacher_recommendation || "Continue comparable probes and review results with the teacher.";
            }
        }

        // IMAGE STROKE DETECTION (If image was attached)
        const imageStrokeBox = document.getElementById("imageStrokeBox");
        if (imageStrokeBox) {
            if (data.file_info && data.file_info.type === "IMAGE" && data.file_info.details && data.file_info.details.annotated_image_url) {
                imageStrokeBox.style.display = "flex";
                const rawImg = document.getElementById("rawSampleImg");
                const annImg = document.getElementById("annotatedSampleImg");
                const statsTxt = document.getElementById("strokeStatsText");
                if (rawImg) rawImg.src = data.file_info.details.raw_image_url + "?t=" + Date.now();
                if (annImg) annImg.src = data.file_info.details.annotated_image_url + "?t=" + Date.now();
                if (statsTxt) {
                    if (data.response_source === "teacher-entered text") {
                        const imageGuess = data.file_info.details.hera_recognized || "no text confidently recognized";
                        statsTxt.textContent = `Teacher-entered response used for analysis. HERA visual suggestion: "${imageGuess}". Review the photo and candidate list.`;
                    } else if (data.file_info.details.hera_recognized) {
                        const candidates = (data.file_info.details.sheet_candidates || []).slice(0, 3).map(c => c.word).join(" / ");
                        statsTxt.textContent = `HERA matched the photo against ${data.file_info.details.sheet_candidate_count} response spellings for this target in the reference sheet. Best match: "${data.file_info.details.hera_recognized}"${candidates ? `; close sheet matches: ${candidates}` : ""}. Please verify.`;
                    } else {
                        statsTxt.textContent = `${data.file_info.details.segmented_boxes} letter outlines detected in sample`;
                    }
                }
            } else {
                imageStrokeBox.style.display = "none";
            }
        }

        // WORD COMPARISON
        const resTarget = document.getElementById("resTargetWord");
        const resTargetSound = document.getElementById("resTargetSound");
        const resSpelled = document.getElementById("resSpelledWord");
        const resSpelledSound = document.getElementById("resSpelledSound");

        if (resTarget) resTarget.textContent = err.target_word || "";
        if (resTargetSound) resTargetSound.textContent = `Spoken: /${err.target_ipa || ""}/`;
        if (resSpelled) resSpelled.textContent = err.spelled_word || "";
        if (resSpelledSound) resSpelledSound.textContent = `Spoken: /${err.spelled_ipa || ""}/`;

        // WHY MISTAKE HAPPENED
        const resExpl = document.getElementById("resSimpleExplanation");
        if (resExpl) resExpl.textContent = err.feedback || "Accurate spelling.";

        // SOUND DIFFERENCES PILLS
        const soundList = document.getElementById("soundDifferencesList");
        if (soundList) {
            soundList.innerHTML = "";
            if (err.differing_articulatory_features && err.differing_articulatory_features.length > 0) {
                err.differing_articulatory_features.forEach(f => {
                    const pill = document.createElement("span");
                    pill.className = "pill pill-sound";
                    pill.textContent = friendlyFeatureName(f);
                    soundList.appendChild(pill);
                });
            } else {
                const pill = document.createElement("span");
                pill.className = "pill";
                pill.textContent = "Exact phonetic match";
                soundList.appendChild(pill);
            }
        }

        // TEACHER TIPS & PRACTICE
        const resLevel = document.getElementById("resPracticeLevel");
        const resQuote = document.getElementById("resPedagogicalQuote");
        if (resLevel) resLevel.textContent = teacher.recommended_tier || "Regular Classroom Support";
        if (resQuote) resQuote.textContent = `"${teacher.educational_quote || ""}"`;

        const actList = document.getElementById("resActivitiesList");
        if (actList && teacher.classroom_drills) {
            actList.innerHTML = "";
            teacher.classroom_drills.forEach(drill => {
                const li = document.createElement("li");
                const parts = drill.split(":");
                if (parts.length > 1) {
                    li.innerHTML = `<strong>${parts[0]}:</strong> ${parts.slice(1).join(":")}`;
                } else {
                    li.textContent = drill;
                }
                actList.appendChild(li);
            });
        }

        // PROGRESS GRAPH & STATS
        const sBadge = document.getElementById("sessionBadge");
        const paceEl = document.getElementById("resLearningPace");
        const slopeEl = document.getElementById("resSlopeText");
        const masteryFill = document.getElementById("masteryFill");
        const masteryPctEl = document.getElementById("resMasteryPct");
        const alertMsg = document.getElementById("resAlertMsg");

        if (sBadge) sBadge.textContent = long.sessions_completed ? `Week ${long.sessions_completed}` : "Week 1";
        const hasTrend = long.verdict !== "INSUFFICIENT_DATA" && long.verdict !== "BASELINE";
        if (paceEl) {
            if (!hasTrend) {
                paceEl.textContent = "Collecting data";
            } else if (long.verdict === "INCREASED") {
                paceEl.textContent = "Error Surge (Regressing)";
            } else if (long.theil_sen_slope < 0) {
                paceEl.textContent = "Improving Velocity";
            } else if (long.theil_sen_slope === 0) {
                paceEl.textContent = "Steady (Plateau)";
            } else {
                paceEl.textContent = "Fluctuating";
            }
        }
        if (slopeEl) {
            if (!hasTrend) {
                slopeEl.textContent = "More comparable probes are needed";
            } else if (long.verdict === "INCREASED") {
                slopeEl.textContent = `Acute surge (+${Math.abs(long.immediate_delta * 100).toFixed(0)}% delta)`;
            } else if (long.theil_sen_slope < 0) {
                slopeEl.textContent = `Error falling by ${(Math.abs(long.theil_sen_slope) * 100).toFixed(1)}% per week`;
            } else {
                slopeEl.textContent = "No downward trend detected";
            }
        }

        const masteryPct = Math.round((long.bkt_mastery_probability ?? 0.2) * 100);
        if (masteryFill) masteryFill.style.width = `${masteryPct}%`;
        if (masteryPctEl) masteryPctEl.textContent = `${masteryPct}% Mastered`;

        if (alertMsg) {
            alertMsg.textContent = long.action_required === "GRADUATE_CONCEPT"
                ? "Mastery achieved! Ready to advance to more complex conjunct letters."
                : (long.action_required === "MODALITY_SHIFT"
                    ? "Plateau noticed. Shifting practice style to hands-on tactile drawing."
                    : (long.action_required === "TIER_ESCALATION"
                        ? "Acute regression spike detected. Recommending Tier 2 intensive support."
                        : "Continuing active monitoring."));
        }

        // DETAILED PSYCHOMETRICS (PhD Pillar 3 - Mechanism N1-N5)
        const metricTheilSen = document.getElementById("metricTheilSen");
        const metricSlopeNote = document.getElementById("metricSlopeNote");
        const metricCusum = document.getElementById("metricCusum");
        const metricCusumNote = document.getElementById("metricCusumNote");
        const metricBkt = document.getElementById("metricBkt");
        const metricAction = document.getElementById("metricAction");
        const metricAuditTrace = document.getElementById("metricAuditTrace");

        if (metricTheilSen) {
            const slopeVal = long.theil_sen_slope ?? 0.0;
            metricTheilSen.textContent = `${slopeVal >= 0 ? "+" : ""}${slopeVal.toFixed(4)} / session`;
            metricTheilSen.style.color = slopeVal < -0.01 ? "var(--success)" : (slopeVal > 0.02 ? "var(--danger)" : "#38bdf8");
        }
        if (metricSlopeNote) {
            metricSlopeNote.textContent = long.theil_sen_slope < -0.01 
                ? "Consistent downward error velocity" 
                : (long.theil_sen_slope > 0.02 ? "Upward error drift (regression)" : "Flat plateau across sessions");
        }
        if (metricCusum) {
            metricCusum.textContent = long.cusum_alert || "NONE";
            metricCusum.style.color = long.cusum_alert === "REGRESSION_SURGE" 
                ? "var(--danger)" 
                : (long.cusum_alert === "BREAKTHROUGH_DROP" ? "var(--success)" : "#94a3b8");
        }
        if (metricCusumNote) {
            metricCusumNote.textContent = long.cusum_alert === "REGRESSION_SURGE" 
                ? "Acute regression spike detected" 
                : (long.cusum_alert === "BREAKTHROUGH_DROP" ? "Statistically significant breakthrough" : "Variance within expected baseline bounds");
        }
        if (metricBkt) {
            const bktPct = Math.round((long.bkt_mastery_probability ?? 0.2) * 100);
            metricBkt.textContent = `${bktPct}%`;
            metricBkt.style.color = bktPct >= 80 ? "var(--success)" : (bktPct >= 40 ? "#38bdf8" : "var(--warning)");
        }
        if (metricAction) {
            metricAction.textContent = long.action_required || "MONITOR";
        }
        if (metricAuditTrace) {
            metricAuditTrace.textContent = long.verified_teacher_report || long.teacher_recommendation || "Claims grounded in deterministic tools (Theil-Sen slope & BKT posterior). Zero LLM hallucination.";
        }

        // REFERENCE DATABASE INFO
        const catName = document.getElementById("resCategoryName");
        const dbFeed = document.getElementById("resDatabaseFeedback");
        const precSrc = document.getElementById("resPrecedentSource");

        if (catName) catName.textContent = err.dimension || "General";
        if (dbFeed) dbFeed.textContent = `"${err.feedback || ""}"`;
        if (precSrc) {
            precSrc.textContent = err.rag_precedent_id !== null 
                ? `Verified case row #${err.rag_precedent_id} in 8,673 teaching corpus`
                : "Direct phonetic sound analysis";
        }

        // Render Chart
        if (chart.labels && chart.error_rates) {
            renderProgressChart(chart.labels, chart.error_rates);
        }
    }

    function friendlyFeatureName(f) {
        const map = {
            "voi": "Vocal cord vibration difference (Voicing)",
            "voice": "Vocal cord vibration difference",
            "ant": "Tongue position forward vs back (Dental / Retroflex)",
            "retroflex": "Tongue curl difference (Retroflex)",
            "delrel": "Air release speed difference (Plosive / Fricative)",
            "cont": "Airflow manner difference (Tap / Trill)",
            "sg": "Breathiness / Aspiration difference",
            "spread_glottis": "Breathiness difference (Aspiration)",
            "long": "Vowel length difference (Short vs Long)",
            "tense": "Vowel stress / tension"
        };
        return map[f] || `Sound feature [${f}]`;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 7. PROGRESS CHART (CHART.JS)
    // ══════════════════════════════════════════════════════════════════════════
    function renderProgressChart(labels, points) {
        const chartCanvas = document.getElementById("progressChart");
        if (!chartCanvas) return;
        const ctx = chartCanvas.getContext("2d");

        if (progressChart) {
            progressChart.destroy();
        }

        progressChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [{
                    label: "Mistake Rate (%)",
                    data: points,
                    borderColor: "#06b6d4",
                    backgroundColor: "rgba(6, 182, 212, 0.12)",
                    borderWidth: 2.5,
                    pointBackgroundColor: "#38bdf8",
                    pointBorderColor: "#ffffff",
                    pointBorderWidth: 1.5,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    fill: true,
                    tension: 0.35
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#0c1527",
                        titleColor: "#f8fafc",
                        bodyColor: "#38bdf8",
                        borderColor: "rgba(56, 189, 248, 0.3)",
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: (c) => `Mistake Rate: ${c.parsed.y}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: "#94a3b8", font: { size: 11 } }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: "rgba(56, 189, 248, 0.08)" },
                        ticks: {
                            color: "#94a3b8",
                            font: { size: 11 },
                            callback: v => v + "%"
                        }
                    }
                }
            }
        });
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 8. RUN HISTORY (PERSISTENCE)
    // ══════════════════════════════════════════════════════════════════════════
    function saveToHistory(data) {
        if (!data || !data.error_analysis || !data.longitudinal_eval) return;

        const entry = {
            id: Date.now(),
            student_name: data.student ? data.student.name : "Arjun K.",
            target_word: data.error_analysis.target_word,
            spelled_word: data.error_analysis.spelled_word,
            verdict: data.longitudinal_eval.verdict,
            time_str: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            full_data: data
        };

        evalHistory.unshift(entry);
        if (evalHistory.length > 20) evalHistory.pop();

        try {
            localStorage.setItem("akshara_history", JSON.stringify(evalHistory));
        } catch (e) {}

        renderHistoryList();
    }

    function loadHistory() {
        try {
            const saved = localStorage.getItem("akshara_history");
            if (saved) {
                evalHistory = JSON.parse(saved);
            } else {
                evalHistory = [];
            }
        } catch (e) {
            evalHistory = [];
        }
        renderHistoryList();
    }

    function renderHistoryList() {
        if (!historyList) return;
        historyList.innerHTML = "";
        if (historyCount) historyCount.textContent = `${evalHistory.length} saved`;

        evalHistory.forEach(item => {
            const div = document.createElement("div");
            div.className = "history-item";
            
            let statusBadgeClass = "hist-status-warn";
            let statusText = "Recorded";
            if (item.verdict === "DECREASED") {
                statusBadgeClass = "hist-status-good";
                statusText = "Improving";
            }
            if (item.verdict === "STAGNANT") {
                statusBadgeClass = "hist-status-warn";
                statusText = "Steady";
            } else if (item.verdict === "INCREASED") {
                statusBadgeClass = "hist-status-bad";
                statusText = "Attention";
            } else if (item.verdict === "BASELINE" || item.verdict === "INSUFFICIENT_DATA") {
                statusBadgeClass = "hist-status-warn";
                statusText = "Data needed";
            }

            const top = document.createElement("div");
            top.className = "hist-top";
            const words = document.createElement("span");
            words.className = "hist-word";
            words.textContent = `${item.target_word} → ${item.spelled_word}`;
            const badge = document.createElement("span");
            badge.className = `hist-status ${statusBadgeClass}`;
            badge.textContent = statusText;
            top.append(words, badge);
            const time = document.createElement("div");
            time.className = "hist-time";
            time.textContent = `${item.student_name} • ${item.time_str}`;
            div.append(top, time);

            div.addEventListener("click", () => {
                if (item.full_data) {
                    renderTeacherReport(item.full_data);
                } else {
                    if (targetWordInput) targetWordInput.value = item.target_word;
                    if (spelledWordInput) spelledWordInput.value = item.spelled_word;
                }
            });

            historyList.appendChild(div);
        });
    }
});
