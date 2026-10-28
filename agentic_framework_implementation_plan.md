# Agentic AI Framework: End-to-End Implementation Plan

**A Phased Engineering & Research Roadmap for Developing, Validating, and Deploying the Longitudinal Multi-Agent Framework for Malayalam Handwriting Error Diagnosis.**

---

## 1. Executive Summary

This implementation plan provides a structured, doctoral-grade engineering roadmap to build the **Agentic AI Framework for Malayalam Handwriting Error Diagnosis and Longitudinal Intervention Tracking**.

The system translates research theories into a working software architecture consisting of:
1. **Dual-Stream Visual Extraction**: SigLIP-So400m + Byte-level CTC / HERA (zero autocorrect bias).
2. **3-Tier G2P Engine**: OSUR-G2P (Neural) + Epitran (Rule-based) + PanPhon (22-D Articulatory Feature Distance).
3. **Longitudinal Memory & Statistical Services**: Theil-Sen Robust Slope, Bayesian Knowledge Tracing (BKT), and CUSUM Control Charts.
4. **Closed-Loop Multi-Session Dynamic Assessment (CMSDA)**: Real-time 3-way progress classification (Decreased / Stagnant / Increased) driving targeted RTI Tier 1–3 interventions.

---

## 2. Technical Stack & Hardware Environment

### 2.1 Core Software Dependencies
| Component | Library / Framework | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `>= 3.10` | Core backend & agent runtime |
| **Deep Learning** | PyTorch / torchvision | `>= 2.2` | Model training & tensor operations |
| **Transformers** | Hugging Face Transformers | `>= 4.38` | SigLIP vision backbone, ByT5 decoder |
| **G2P (Neural)** | OSUR-G2P / Fairseq / PyTorch | custom | Learner-error-preserving G2P |
| **G2P (Rule)** | Epitran (`epitran.Epitran('mal-Mlym')`) | `>= 0.70` | Deterministic transliteration baseline |
| **Articulatory Features**| PanPhon (`panphon.FeatureTable()`) | `>= 0.20` | 22-dimensional phonetic feature distance |
| **Statistics** | SciPy (`scipy.stats.theilslopes`), NumPy | `>= 1.12` | Non-parametric robust linear regression |
| **Agent Runtime** | LangGraph / Autogen / LiteLLM | `>= 0.1` | State machine orchestration & self-critique loop |
| **Vector Database** | ChromaDB / FAISS | `>= 0.4` | RAG over 8,674-row expert error taxonomy |
| **Persistence** | SQLite / PostgreSQL | `>= 15` | Longitudinal time-series session memory |

### 2.2 Hardware Requirements
* **Development/Inference**: NVIDIA RTX 3090/4090 (24GB VRAM) or Apple Silicon M2/M3 Max (36GB+ Unified Memory).
* **Quantized Deployment**: Intel Core i7/i9 or AMD Ryzen 9 (16GB RAM) running 4-bit/8-bit quantized SigLIP and ByT5 via ONNX Runtime / `bitsandbytes`.

---

## 3. Phased Implementation Roadmap

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             ENGINEERING TIMELINE                                 │
│                                                                                  │
│  [Month 1] Phase 1: Dual-Stream OCR (SigLIP + HERA) & Data Curation             │
│  [Month 2] Phase 2: 3-Tier G2P & PanPhon Articulatory Engine Integration        │
│  [Month 3] Phase 3: Longitudinal Statistical Services (Theil-Sen, BKT, CUSUM)   │
│  [Month 4] Phase 4: Multi-Agent Orchestration & Self-Verification Critique Loop  │
│  [Month 5] Phase 5: Closed-Loop CMSDA Protocol & Teacher Verification Gate       │
│  [Month 6] Phase 6: Clinical Validation, Empirical Ablations & Thesis Synthesis │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Dual-Stream OCR & Grapheme Preservation (Month 1)

#### Objective
Ensure handwriting extraction captures the child's exact visual marks without language-model autocorrection.

#### Tasks
1. **Task 1.1: Image Preprocessing Pipeline**
   * Implement automated page deskewing, line segmentation, and Akshara bounding-box detection using Otsu binarization and morphological contour dilation.
   * Standardize bounding-box crops to $448 \times 448$ RGB tensors.
2. **Task 1.2: SigLIP-So400m Feature Extractor Setup**
   * Load pretrained `google/siglip-so400m-patch14-384` / `google/siglip-base-patch16-224`.
   * Freeze the vision backbone and attach a character-level CTC (Connectionist Temporal Classification) projection head over Malayalam Unicode code points.
   * Fine-tune using LoRA (rank=16, alpha=32) on the HP-Malayalam child handwriting split.
3. **Task 1.3: HERA kNN Exemplar Bridge**
   * Build the kNN vector index over HERA's prototype representations of valid and deformed Malayalam Aksharas.
   * Implement the dual-stream voting arbiter: resolve character identities verbatim, preserving irregular ligatures and erroneous chillu endings.

#### Deliverable
* `src/vision/siglip_extractor.py` & `src/vision/hera_matcher.py` passing Character Error Rate (CER) benchmarks on misspelled words with **0% autocorrection drift**.

---

### Phase 2: 3-Tier G2P & PanPhon Articulatory Engine (Month 2)

#### Objective
Convert extracted verbatim text to IPA, cross-validate confidence, and compute fine-grained articulatory feature vectors.

#### Tasks
1. **Task 2.1: OSUR-G2P Model Wrapper (Tier 1)**
   * Deploy the neural sequence-to-sequence model trained on the 3,800+ child error corpus.
   * Validate Phone Error Rate (PER $\le 0.85\%$) on non-word evaluation benchmarks.
2. **Task 2.2: Epitran Transliteration Wrapper (Tier 2)**
   * Configure `epitran.Epitran('mal-Mlym')` with custom pre/post-FST rules for chillu characters and samvrithokaram.
3. **Task 2.3: Cross-Validation Confidence Gate**
   * Implement `cross_validate(osur_ipa, epitran_ipa)`:
     * Label as `AGREE` if edit distance $= 0$.
     * Label as `MINOR_DIFF` if distance $\le 2$ (contextual vowel rules; trust OSUR).
     * Label as `MAJOR_DIFF` if distance $> 2$ (flag for human-in-the-loop audit).
4. **Task 2.4: PanPhon 22-D Articulatory Feature Decomposition (Tier 3)**
   * Build `src/phonology/panphon_engine.py` using `panphon.FeatureTable()`.
   * Implement the **Weighted Feature Distance (WFD)** calculator:
     $$\text{WFD}(p_1, p_2) = \sum_{i=1}^{22} w_i \cdot |f_i(p_1) - f_i(p_2)|$$
   * Map specific Malayalam phonetic confusion pairs (ക/ഗ, ര/റ, ല/ള, ന/ണ) to clinical feature differentials (`[±voice]`, `[±continuant]`, `[±retroflex]`).

#### Deliverable
* `src/phonology/three_tier_g2p.py` producing calibrated IPA transcripts and articulatory deltas for arbitrary Malayalam strings.

---

### Phase 3: Longitudinal Memory & Statistical Engines (Month 3)

#### Objective
Construct time-series storage and deterministic statistical tools to evaluate longitudinal child trajectories.

#### Tasks
1. **Task 3.1: Relational Schema & Memory Store**
   * Create database tables:
     * `Students` (id, age, grade, language_background)
     * `AssessmentSessions` (session_id, student_id, timestamp, intervention_tier)
     * `ExtractedItems` (item_id, target_word, spelled_word, error_type, wfd_score)
     * `LongitudinalMetrics` (session_id, dimension, error_rate, bkt_mastery)
2. **Task 3.2: Theil-Sen Robust Trend Service**
   * Implement `calculate_theil_sen_slope(timestamps, error_rates)`.
   * Compute 95% confidence intervals and p-values to reliably measure true learning velocity $\beta$ despite outlier "bad days".
3. **Task 3.3: Bayesian Knowledge Tracing (BKT) Engine**
   * Implement the standard 4-parameter BKT model ($P(L_0)=0.2, P(T)=0.15, P(G)=0.1, P(S)=0.05$).
   * Track latent mastery probability $P(L_t)$ per orthographic dimension across successive assessments.
4. **Task 3.4: CUSUM Change-Point Detector**
   * Implement sequential cumulative sum testing ($S_t^+, S_t^-$) with allowance $k=0.5\sigma$ and threshold $h=4\sigma$.
   * Auto-trigger flags for breakthrough milestones or acute regression spikes.

#### Deliverable
* `src/stats/longitudinal_engine.py` unit-tested on simulated and real student time-series profiles.

---

### Phase 4: Multi-Agent Orchestration & Self-Verification (Month 4)

#### Objective
Build the agent layer where LLM reasoning is strictly separated from tool execution, backed by a self-verifying critique-revise loop.

#### Tasks
1. **Task 4.1: Agent Definitions & Tool Registries**
   * Implement the **Error-Analysis Agent**: reads visual crops, calls G2P tools, searches Gold DB RAG, produces triple-grounded classifications.
   * Implement the **Insight Agent**: queries longitudinal memory, calls stats tools, interprets trend slopes.
   * Implement the **Intervention Planner Agent**: maps error dimensions to targeted Malayalam phonics worksheets and minimal pairs.
2. **Task 4.2: LangGraph Orchestration State Graph**
   * Define typed state schema: `AgentState` containing student metadata, image paths, tool returns, and proposed reports.
   * Configure deterministic tool calling nodes with JSON schema validation.
3. **Task 4.3: Self-Verification Critique-Revise Loop**
   * Implement the assertion validator: check that every statement in the draft insight (e.g., *"Child is regressing in dental/retroflex distinction"*) is supported by a positive Theil-Sen slope ($\beta > 0, p < 0.05$).
   * Automatically rewrite or eliminate unsupported claims before presenting output to teachers.

#### Deliverable
* `src/agents/orchestrator.py` executing full end-to-end multi-agent runs with deterministic self-correction logging.

---

### Phase 5: Closed-Loop CMSDA Protocol & Teacher Gate (Month 5)

#### Objective
Deliver the closed-loop evaluation system that continuously responds whenever new data is added by checking if errors have decreased, stayed stagnant, or increased.

#### Tasks
1. **Task 5.1: 3-Way Closed-Loop Decision Engine**
   * Codify the decision matrix:
     * **Decreased**: $\beta < -0.02, \Delta < -0.08 \implies$ Graduate skill, advance to next Akshara complexity level.
     * **Stagnant**: $|\beta| \le 0.02, |\Delta| \le 0.05 \implies$ Trigger Modality Shift (e.g., transition from worksheets to multi-sensory clay/sand-tray tracing).
     * **Increased**: $\beta > +0.02, \Delta > +0.08 \implies$ Escalate RTI Tier (Tier 1 regular $\to$ Tier 2 targeted phonics $\to$ Tier 3 clinical evaluation).
2. **Task 5.2: Teacher Verification Gate & UI Dashboard**
   * Build an interactive Streamlit or web UI:
     * Display original handwritten crop alongside extracted graphemes and PanPhon articulatory radar plots.
     * Present the longitudinal trend chart with Theil-Sen slope and CUSUM change points.
     * Provide a one-click **"Approve Intervention"** button allowing the educator to override, adjust, or confirm the suggested pedagogical plan.
3. **Task 5.3: Automated Probe Word List Generator**
   * Query the 13,577-word Malayalam lexicon for targeted minimal pairs focusing on the student's active error locus (e.g., generate *മരം/മറം*, *വില/വിള* probes for next week's test).

#### Deliverable
* Working web dashboard and closed-loop re-assessment cycle tested on pilot classroom scenarios.

---

### Phase 6: Empirical Validation, Ablation Studies & Thesis Writing (Month 6)

#### Objective
Execute rigorous experimental benchmarks to support doctoral defense claims and finalize publication manuscripts.

#### Tasks
1. **Task 6.1: Comprehensive System Ablation Studies**
   * Compare:
     * Full Framework vs. Single-Tool (Epitran alone vs. 3-Tier ensemble).
     * SigLIP-ByGrapheme vs. Standard Tesseract / Google Vision OCR (autocorrect error rate).
     * Theil-Sen vs. Ordinary Least Squares (OLS) regression on outlier-injected student profiles.
2. **Task 6.2: Clinical Concordance Evaluation**
   * Calculate Cohen's Kappa ($\kappa$) agreement between the Agentic framework's error classifications and certified Malayalam special educators / speech-language pathologists.
3. **Task 6.3: Doctoral Documentation & Publication Drafting**
   * Finalize the PhD dissertation chapters.
   * Prepare conference/journal submissions (e.g., IEEE/ACM Transactions on Learning Technologies or Interspeech).

---

## 4. Evaluation Metrics & Success Criteria

| Evaluation Dimension | Metric | Target Threshold | Clinical / Academic Rationale |
| :--- | :--- | :--- | :--- |
| **Grapheme Extraction** | Character Error Rate (CER) on misspelled data | $\le 4.5\%$ | Must extract raw handwriting accurately without autocorrect |
| **Error Preservation** | Autocorrect Distortion Rate | **$0.0\%$** | Crucial: A single autocorrect destroys dysgraphia/dyslexia diagnosis |
| **G2P Accuracy** | Phone Error Rate (PER) | $\le 0.85\%$ | Accurate phoneme mapping across arbitrary learner pseudo-words |
| **Diagnostic Agreement** | Cohen's Kappa ($\kappa$) vs. Human Experts | $\kappa \ge 0.88$ | Clinical reliability for special education recommendations |
| **Hallucination Rate** | Critique-Revise Unsupported Assertion Rate | **$0.0\%$** | Self-verification loop must catch all ungrounded LLM claims |
| **Longitudinal Power** | Theil-Sen Outlier Breakdown Point | $\ge 29.0\%$ | Resilient against random "bad test days" |

---

## 5. Risk Assessment & Mitigation Strategy

| Risk Factor | Severity | Probability | Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **Extreme child handwriting variability / stroke overlap** | High | Medium | Dual-stream redundancy (SigLIP visual embeddings + HERA prototype kNN); low-confidence flag triggers manual teacher verification. |
| **Non-standard colloquial dialect pronunciations** | Medium | Medium | Maintain regional phonetic variation tables in OSUR-G2P; use PanPhon WFD to treat minor regional phonetic shifts as low severity. |
| **Over-reliance on LLM for diagnostic decisions** | Critical | Low | Strictly enforced Tool vs. Agent separation: all classifications, trend calculations, and mastery metrics are computed deterministically. |
| **Teacher friction with complex technical reports** | Medium | Medium | Scaffold reports into two layers: a clean, actionable 3-line teacher summary and an expandable clinical evidence appendix. |

---

## 6. Immediate Next Steps

1. **Verify Local Environment**: Initialize Python 3.10 virtual environment and install PyTorch, Transformers, Epitran, and PanPhon.
2. **Dataset Partitioning**: Extract and index the 8,674-row Malayalam error classification dataset into ChromaDB/FAISS.
3. **Run Baseline Benchmark**: Test SigLIP-So400m against a pilot batch of Malayalam child handwriting crops to calibrate the character extraction head.
