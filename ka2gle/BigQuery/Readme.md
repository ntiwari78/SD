# BigQuery AI Hackathon — 6 Winning-Grade Project Blueprints (3 per category)


## Executive Summary

The **BigQuery AI Hackathon** is a Kaggle-hosted, Google Cloud–backed event focused on building with **BigQuery AI (ML.GENERATE\_TEXT / AI.GENERATE, ML.GENERATE\_EMBEDDING), BigQuery ML, and BigQuery Vector Search**, often combined with Vertex AI for models and orchestration. It runs for \~6 weeks (kicked off **Aug 11, 2025 UTC ≈ Aug 11, 2025 IST**) with multiple tracks (Generative AI, Vector Search, Multimodal) and a published prize pool (USD \$100k) emphasizing real-world utility and clarity of demos. ([Google Cloud][1])

Treat the **Kaggle competition Overview/Rules/Discussions** as ground truth for format, eligibility, use of external data, and deliverables; when details are ambiguous (e.g., what external data is allowed), confirm on the **Rules** and **Discussions** tabs. ([Kaggle][2])

Your winning edge: (1) **ship a working demo** that judges can click; (2) **leverage BigQuery-native AI** (vector search + BQML + remote Vertex models) with **explainability**; (3) do **time-aware evaluation** (for Finance) and **safety + adaptivity** (for EdTech); (4) show **cost-aware SQL** and **clear artifacts** (Looker Studio dashboards, model cards, vector-index explainers). For text + vector work, use **BigQuery Vector Search** and **ML.GENERATE\_EMBEDDING**; for RAG, use Vertex AI embeddings with **BigQuery remote models**. ([Google Cloud][3])



# Category A — Companies’ Financial Health from Quarterly Results (3 ideas)

### Idea A1 — **“4Q Distress Watch”**

**One-liner:** Predict a company’s **12-month (next 4 quarters) distress probability**, blending financial ratios with MD\&A risk-factor signals.

**1) Problem Definition & Target**

* **Target:** Binary classification — **distress within 4 quarters** (proxy: going-concern warning, negative OCF in ≥2 of next 4 quarters, breach-like leverage spikes, or Altman-Z < threshold).
* **Horizon & Granularity:** **Quarterly**, rolling 4Q prediction window.

**2) Data & Features**

* **Primary:** Parsed **10-Q/10-K** (revenue, margins, OCF/FCF, debt, current ratio, accruals), **YoY/QoQ deltas**, **Altman-Z components**, **accruals quality** (ΔWC − ΔCash − ΔSTDebt vs CFO).
* **Text (optional):** MD\&A “Risk Factors” embeddings from filings (EDGAR). Hugging Face **EDGAR-CORPUS** (Apache-2.0) can seed MD\&A sectioning and patterns; **JanosAudran/financial-reports-sec** (Apache-2.0) provides report-level content for parsing. ([Hugging Face][4])
* **Public macro:** CPI, rates as controls (e.g., BLS CPI, FRED Funds Rate). If using public BigQuery datasets/Marketplaces (BLS CPI in Marketplace), cite them in your notebook and join in BQ. ([Google Cloud][5], [Google Cloud][1], [FRED][6])
* **BigQuery-native:** Land filings in **BigQuery** (GCS → BQ). Use **ML.GENERATE\_EMBEDDING** (via Vertex AI remote model) to embed MD\&A chunks and store vectors for *text-tabular fusion*. ([Google Cloud][7])

**3) Evaluation & Success Criteria**

* **Metrics:** **PR-AUC** (class imbalance), **AUROC** secondary.
* **CV:** **Purged, grouped time split** by **ticker**, blocking leakages across adjacent quarters (no look-ahead).

**4) Architecture on GCP**

* **Feature factory:** BigQuery SQL (window/lag/lead, SAFE\_DIVIDE).
* **Model:** **BQML XGBoost** classifier baseline; consider **AutoML Tables via BQML** for quick tuning. ([Google Cloud][8])
* **Text fusion:** For MD\&A features, either (a) average embeddings by quarter; or (b) top-k **VECTOR\_SEARCH** matches per risk prompt and turn them into numeric features (counts/similarity scores). ([Google Cloud][3])

**5) MVP in 24–48h (steps)**

* **SQL skeleton (features):**

```sql
CREATE OR REPLACE TABLE feat.fin_quarter AS
SELECT
  cik, ticker, filing_qtr, filing_dt,
  revenue, gross_margin, op_margin, net_margin,
  SAFE_DIVIDE(operating_cash_flow, revenue) AS ocf_margin,
  total_debt, SAFE_DIVIDE(total_debt, ebitda) AS leverage,
  current_ratio,
  revenue - LAG(revenue) OVER w AS qoq_rev_delta,
  SAFE_DIVIDE(revenue - LAG(revenue) OVER w, NULLIF(LAG(revenue) OVER w,0)) AS qoq_rev_pct,
  SAFE_DIVIDE(revenue - LAG(revenue,4) OVER w, NULLIF(LAG(revenue,4) OVER w,0)) AS yoy_rev_pct,
  -- Altman-like components, accruals quality placeholders...
FROM raw.financials
WINDOW w AS (PARTITION BY ticker ORDER BY filing_qtr);
```

* **Embeddings (remote model + BQML):** create remote model (Vertex **text embeddings**), then:

```sql
CREATE OR REPLACE MODEL models.vertex_embed
REMOTE WITH CONNECTION `region.conn`
OPTIONS (endpoint = 'text-embedding-005');  -- example

CREATE OR REPLACE TABLE feat.mda_embed AS
SELECT
  ticker, filing_qtr,
  ML.GENERATE_EMBEDDING(MODEL `models.vertex_embed`,
    (SELECT 'content' AS content, mda_text)) AS emb
FROM raw.mda_chunks;
```

* **Vector index (optional):** `CREATE VECTOR INDEX idx_mda ON feat.mda_embed(emb) OPTIONS(index_type='IVF');` ([Google Cloud][9])
* **Train (BQML):**

```sql
CREATE OR REPLACE MODEL models.distress_xgb
OPTIONS (model_type='XGBOOST_CLASSIFIER', input_label_cols=['will_distress'], enable_global_explain=true)
AS
SELECT * FROM feat.training_view;  -- time-aware filter
```

* **Evaluate:** `SELECT * FROM ML.EVALUATE(MODEL models.distress_xgb);` (record PR-AUC/AUROC). ([Google Cloud][10])

**6) Risks & Mitigation**

* **Leakage / look-ahead:** strict quarter cutoffs, purged gaps.
* **Survivorship bias:** include delisted tickers where possible.
* **Licenses:** Only use **open-licensed** corpora (Apache-2.0 OK); log any non-commercial sources as “not used in final”. ([Hugging Face][4])

**7) Why This Could Win**
Strong **tabular + text fusion** within **BigQuery** (no extra infra), **transparent CV**, and **Explainability (BQML global feature importance)** for judges. ([GitHub][11])

**8) Artifacts to Demo**
Looker Studio dashboard (distress heatmap & trend), PR-AUC chart, top features, per-company risk card, quick **Vector Search** explainer SQL snippet. ([Google Cloud][12])


### Idea A2 — **“Cash-Runway Radar”**

**One-liner:** Regress **months of cash runway** and flag firms with **<12 months** runway, updated quarterly.

**1) Problem & Target**

* **Target (regression + threshold):** `RunwayMonths = Cash / max(0, OperatingExpense - NonCashAdjustments)`; alert class if `< 12`.
* **Horizon:** next **4 quarters**.

**2) Data & Features**

* Financial statements (cash, opex, SBC, interest, capex), **burn rate trends**, seasonality flags, sector dummies; optional macro (rates, CPI). **BLS CPI (Marketplace)** and **FRED rate series** are standard macro joins. ([Google Cloud][5], [FRED][6])

**3) Evaluation**

* **Regression:** **SMAPE/MAE**; **Classification:** PR-AUC for `<12m` alerts.
* **CV:** forward-chaining by ticker.

**4) Architecture**

* Feature SQL in BigQuery; model with **BQML regression** (linear/XGBoost) + **calibration** table for alert threshold. ([Google Cloud][8])

**5) 24–48h MVP**

* Compute quarterly **burn** and runway via SQL; train **XGBOOST\_REGRESSOR**; create a view for **red/yellow/green** alerts; embed in Looker Studio.

**6) Risks**

* **Capex/one-off distortions:** winsorize; use rolling medians.
* **Sparse disclosures:** fallback imputation rules.

**7) Why It Could Win**
Clear business value, **intuitive metric (months)**, easy to demo on a dashboard.

**8) Artifacts**
Runway distribution, alert list, SHAP-like importance (global explain). ([GitHub][11])


### Idea A3 — **“Earnings-Quality Index (EQI)”**

**One-liner:** A **composite financial-health score** emphasizing **quality of earnings** (accruals, cash conversion, margins stability) and **MD\&A clarity**.

**1) Problem & Target**

* **Target:** **0–100 index**, quantile-ranked by sector each quarter.
* **Horizon:** **Quarterly** score with 4Q smoothing.

**2) Data & Features**

* **Cash conversion cycle**, **accruals quality** (Dechow-Dichev style proxies), **margin stability**, **revenue persistence**; **text clarity** from MD\&A via embeddings and **similarity to “clear writing” exemplars** (Vector Search scoring). ([Google Cloud][12])

**3) Evaluation**

* Offline: correlation of **EQI(t)** with **distress next 4Q (AUROC/PR-AUC)** and **future volatility**; also **rank metrics** (Spearman).

**4) Architecture**

* BQ feature SQL; **BQML AutoML Tables (regression)** to predict forward stability → converted to index; Vector Search to compute MD\&A clarity scores using **ML.GENERATE\_EMBEDDING**. ([Google Cloud][7])

**5) 24–48h MVP**

* Build index components in SQL, z-score normalize, sector-neutral combine; validate monotonic ties with forward risk.

**6) Risks**

* **Sector heterogeneity:** sector-neutral ranking.
* **Text noise:** chunk MD\&A by headings; store chunk embeddings.

**7) Why It Could Win**
Judges love **interpretable composite scores** + **vector-powered text insight** all **inside BigQuery**.

**8) Artifacts**
Index methodology card, sector heatmaps, per-company EQI report.

---

### Comparison — Category A

| Idea                   | Target & Horizon                | Core Features                              | Model Path (BQML/Vertex)             | Metric                     | Risk Level         | Differentiator                 | 48h Feasibility |
| ---------------------- | ------------------------------- | ------------------------------------------ | ------------------------------------ | -------------------------- | ------------------ | ------------------------------ | --------------- |
| 4Q Distress Watch      | Distress in next 4Q (quarterly) | Ratios, deltas, Altman-Z, MD\&A embeddings | **BQML XGBoost** + **Vector Search** | PR-AUC, AUROC              | **Med** (labeling) | Tabular+text fusion, robust CV | **High**        |
| Cash-Runway Radar      | Months of runway + `<12m` alert | Cash, burn, opex trends, macro             | **BQML Regressor/XGB**               | SMAPE/MAE; PR-AUC          | Low-Med            | Business-friendly KPI          | **High**        |
| Earnings-Quality Index | 0–100 quality score (quarterly) | Accruals quality, stability, MD\&A clarity | **AutoML via BQML** + embeddings     | Spearman, PR-AUC (linkage) | Med                | Composite EQ + explainability  | **Med-High**    |


# Category B — Intelligent Science Tutor (3 ideas)

*All ideas: explain → formative quiz → diagnose misconceptions → remediate → checkpoint test. Retrieval uses **Vertex AI Embeddings** or **ML.GENERATE\_EMBEDDING** with **BigQuery Vector Search**; state is stored in **BigQuery tables**. Safety: cite-first answers (source paragraphs), profanity filter, “no medical/legal” guardrails.* ([Google Cloud][13])

### Idea B1 — **“MechMentor Lite” (Physics: Mechanics, class 9–10)**

**Scope & Pedagogy:** Kinematics → Newton’s laws → Work-Energy → Momentum. Explanations with **OpenStax Physics** (CC-BY-4.0), quizzes bootstrapped from **AI2-ARC (easy/medium)**. ([Hugging Face][14])

**Content & Data (HF)**

* **andrewmvd/openstax-textbooks** (CC-BY-4.0, 51 books; text) — primary content. ([Hugging Face][14])
* **allenai/ai2\_arc** (CC-BY-SA-4.0, \~7.8k Qs; text QA) — quiz seed bank (curate physics items).

**Adaptivity Logic**

* **Skill map:** {Kinematics, Forces, Energy, Momentum}.
* **IRT-lite:** difficulty tags from item stats; **Bayesian Knowledge Tracing** per skill.

**Architecture on GCP**

* Chunk OpenStax → **ML.GENERATE\_EMBEDDING** into BQ vectors; **VECTOR\_SEARCH** for retrieval by skill; prompt Vertex **ML.GENERATE\_TEXT** remote model using retrieved chunks; student state in BQ tables. ([Google Cloud][7])

**Evaluation & Success**

* **Learning gains:** Pre/post 8-item test Δ; correctness ≥80% on checkpoints; **citation rate** ≥90%.

**24–48h MVP**

* Ingest chapters (mechanics), chunk to 800-1,000 chars, embed, 25 curated questions; wire a simple adaptive loop (pseudocode later).

**Risks & Mitigation**

* **License compliance:** OpenStax CC-BY with attribution (OK).
* **Hallucination:** require **retrieval-first**; show citations.

**Demo & UX (90s)**

1. Select “Newton’s 2nd Law”, 2) short explanation, 3) ask MCQ, 4) student errs (units), 5) system diagnoses “unit confusion”, 6) micro-lesson on N=kg·m/s², 7) new item, 8) mastery badge.


### Idea B2 — **“ChemCoach: Stoichiometry Sprint” (Chemistry, class 10)**

**Scope & Pedagogy:** Mole concept → balancing equations → stoichiometric ratios → limiting reagents.
**Data:** OpenStax **Chemistry 2e** (in OpenStax dataset); **SciQ** (science QA) can seed items but note **CC-BY-NC-3.0** (non-commercial) — flag and avoid in final if rules require fully open commercial use. ([Hugging Face][14])

**Adaptivity:** misconception bank (moles vs mass, rounding), **difficulty ladder** per subskill.

**Architecture:** as in B1; **worked solution generator** with **ML.GENERATE\_TEXT** constrained by retrieved steps. ([Google Cloud][15])

**Evaluation:** pre/post quiz Δ; **worked-step accuracy** (manual rubric for demo).

**Risks:** **SciQ non-commercial** — default to OpenStax only if Kaggle rules require. ([Hugging Face][16])


### Idea B3 — **“BioCell Scout” (Biology: Cell structure & function, class 9–10)**

**Scope:** Cells → organelles → membranes → transport.
**Data:** OpenStax **Biology 2e** (CC-BY-4.0), **allenai/quartz** (science explanation QA; MIT license) for reasoning-style prompts. ([Hugging Face][14])

**Adaptivity:** concept graph (organelles → functions), **confusion triggers** (mitochondria vs chloroplasts), **BKT** per node.

**Architecture:** OpenStax chunks in BQ vectors, **VECTOR\_SEARCH** retrieval, **checkpoint batch** logged to BQ; mastery gates unlock summaries. ([Google Cloud][12])

**Evaluation:** Δ mastery per node; **factual consistency** via citation checks.


### Comparison — Category B

| Idea            | Subject/Grade | Data Source(s) & License                                  | Adaptivity Method    | Retrieval Strategy                 | Safety Plan                        | 48h Feasibility          | Differentiator                               |
| --------------- | ------------- | --------------------------------------------------------- | -------------------- | ---------------------------------- | ---------------------------------- | ------------------------ | -------------------------------------------- |
| MechMentor Lite | Physics 9–10  | OpenStax (CC-BY-4.0), AI2-ARC (CC-BY-SA-4.0)              | BKT + IRT-lite       | BQ Embeddings + **VECTOR\_SEARCH** | Citation-first, profanity filter   | **High**                 | Focused mechanics path + unit misconceptions |
| ChemCoach       | Chemistry 10  | OpenStax (CC-BY-4.0); **SciQ (CC-BY-NC-3.0)** — tentative | BKT + misconceptions | Same                               | Exclude NC data in final if needed | **High** (OpenStax-only) | Worked-solutions scaffold                    |
| BioCell Scout   | Biology 9–10  | OpenStax (CC-BY-4.0), **quartz (MIT)**                    | BKT + concept graph  | Same                               | Citation-first, grade level checks | **High**                 | Visual organelle cards + mastery gates       |



## Hugging Face Dataset Scan & “Similar Topics” Suggestions

### Finance/Markets Shortlist

| Dataset                         | Link               |        License |               Size | Modality     | Why Relevant                                                                                         |
| ------------------------------- | ------------------ | -------------: | -----------------: | ------------ | ---------------------------------------------------------------------------------------------------- |
| **EDGAR-CORPUS**                | Hugging Face       | **Apache-2.0** | 220k+ docs (split) | Text         | Annual report sections (MD\&A/Risk) for MD\&A embeddings/features. ([Hugging Face][4])               |
| **financial-reports-sec**       | Hugging Face       | **Apache-2.0** |           (varies) | Text         | SEC report texts (more recent); easy parsing and sectioning. ([Hugging Face][17])                    |
| **SubjECTive-QA (earnings QA)** | Hugging Face       |  **CC-BY-4.0** |          2,747 QAs | Text/Tabular | Subjectivity labels for answers in earnings calls → sentiment/clarity features. ([Hugging Face][18]) |
| **SPGISpeech**                  | HF (requires auth) |    Kensho EULA |      5,000 h audio | Audio/Text   | Speech side of earnings calls (ASR experiments) — **license constraints**. ([Hugging Face][19])      |
| **Bloomberg Financial News**    | HF                 |     Apache-2.0 |             \~10k+ | Text         | News features around quarters (event controls); check date coverage. ([Hugging Face][20])            |

> Some finance sets (e.g., **Financial PhraseBank**) have **restrictive/unclear licenses**; handle with care. ([Hugging Face][21])

### K-12 Science/Tutoring Shortlist

| Dataset                | Link         |          License |     Size | Modality | Why Relevant                                                                                |
| ---------------------- | ------------ | ---------------: | -------: | -------- | ------------------------------------------------------------------------------------------- |
| **openstax-textbooks** | Hugging Face |    **CC-BY-4.0** | 51 books | Text     | Textbook-grade content (Physics/Chem/Bio). ([Hugging Face][14])                             |
| **AI2-ARC**            | Hugging Face | **CC-BY-SA-4.0** |   \~7.8k | Text QA  | Science MCQs for formative quizzes.                                                         |
| **quartz**             | Hugging Face |          **MIT** |   \~2.7k | Text QA  | Explanatory QA for reasoning scaffolds. ([Hugging Face][4])                                 |
| **SciQ**               | HF           | **CC-BY-NC-3.0** |    \~13k | Text QA  | Useful for item seeds, but **non-commercial** → avoid if rules forbid. ([Hugging Face][16]) |

### Extra Project Topics (rule-compatible drafts)

**(A) Stock-Intelligence:**

1. **Earnings Call Clarity Score** — Rate clarity/specificity of management answers using **SubjECTive-QA** labels; correlate with next-quarter errors/volatility. ([Hugging Face][18])
2. **MD\&A Change Detector** — Compare current vs prior MD\&A embeddings from **EDGAR-CORPUS**; flag large semantic drift as a risk feature. ([Hugging Face][4])

**(B) K-12 Tutor Companions:**

1. **Worked-Examples Generator** — Use **OpenStax** text to generate and grade worked steps (LLM + retrieved policies). ([Hugging Face][14])
2. **Concept Drift Alert for Syllabus** — Track student cohort errors; adapt lesson order (skill map stored in BQ).



## Rules & Compliance Check (quote/point to critical clauses)

* **Kaggle Overview/Rules/Discussions** are the source of truth for: eligibility, judging, and **whether external data & pretrained models are allowed**. The Rules page must be reviewed after login/acceptance (content not publicly scrapeable here). **Assumption:** Many Kaggle hackathons **permit external, publicly available data if licensed and cited**, but **you must confirm** on this competition’s **Rules** and/or ask in **Discussions**. ([Kaggle][22])
* **Ambiguities & How to Verify (action items):**

  * *External datasets (HF/OpenStax/SEC)* — post a clarification in **Kaggle Discussions** linking to licenses; proceed only with **open commercial-friendly** sources if required (e.g., **Apache-2.0, CC-BY, MIT**). ([Kaggle][23])
  * *Use of Vertex AI via BigQuery ML remote models* — allowed since the hackathon centers on BigQuery AI; ensure queries comply with project quotas and rate limits (e.g., **ML.GENERATE\_TEXT** concurrency). ([Google Cloud Community][24])
  * *Google Cloud credits* — see pinned thread/announcements for credit access (if applicable). ([Kaggle][25])



## Cross-Validation, Metrics & Offline–Online Parity

**(A) Finance**

* **Time-series aware CV:** for each fold, use quarters ≤ *t* for train and *t+1* for validation; group by ticker to avoid entity leakage.
* **Leakage traps:** future macro joins; restatements; re-indexing.
* **Reproduce metric locally:**

  * Distress (A1): **PR-AUC** primary, **AUROC** secondary.
  * Runway (A2): **SMAPE/MAE**; also thresholded alerts with PR-AUC.
  * EQI (A3): **Spearman** rank corr.

**BQML Training Snippet (i):**

```sql
CREATE OR REPLACE MODEL models.fin_xgb
OPTIONS (model_type='XGBOOST_CLASSIFIER',
         input_label_cols=['label'],
         enable_global_explain=TRUE)
AS
SELECT * FROM feat.train_where_quarter <= '2024Q4';
```

([Google Cloud][8])

**(B) Tutor Evaluation & Safety**

* **Rubric:** clarity (1–5), adaptivity (1–5), correctness (citation-supported), grade appropriateness, refusal for unsafe queries.
* **Automatic checks:**

  * **Reading level** constraints in prompt;
  * **Retrieval-citation rate** ≥90%;
  * **Refusal rate** on disallowed topics;
  * **Pre/Post delta** on a fixed 10-item test.

**Adaptive Controller Loop (ii) — Pseudocode**

```python
state = get_student_state(student_id)  # skill_mastery: dict(skill->p)
skill = select_focus_skill(state)      # lowest mastery with prerequisites met
ctx = vector_search(skill, k=4)        # BigQuery VECTOR_SEARCH on OpenStax chunks
explain = llm_generate_text(ctx, prompt="teach briefly + example")  # ML.GENERATE_TEXT
item = pick_question(skill, target_difficulty=adaptive_target(state[skill]))
resp = ask(item)
score = evaluate(resp, item.answer)
state = update_bkt(state, skill, score)  # Bayesian update
if misconception_tag(score): enqueue_remediation(skill, tag)
log_interaction(student_id, skill, item_id, score, ctx.citations)
repeat until mastery >= threshold or time_up
```

([Google Cloud][12])



## GCP Architecture Blueprints (both categories)

**Data Flow (text diagram):**
**Ingest** (EDGAR/OpenStax/your CSV) → **Cloud Storage** → **BigQuery (tables)** → **(Optional) BigQuery remote model to Vertex AI for embeddings/text** → **BigQuery Vector Search** (CREATE VECTOR INDEX + VECTOR\_SEARCH) → **Modeling** (BQML XGBoost/AutoML) → **Serving** (**Cloud Run** HTTPS API calling BQ) → **Dashboard** (Looker Studio / BigQuery Studio saved queries). ([Google Cloud][7])

**Terraform-lite / gcloud checklist (indicative):**

* `gcloud services enable bigquery.googleapis.com aiplatform.googleapis.com`
* Create BQ dataset: `bq --location=US mk -d finhack`
* Create remote model for embeddings: (as in docs) `CREATE OR REPLACE MODEL ... REMOTE WITH CONNECTION ...` ([Google Cloud][7])
* Generate embeddings → store in BQ;
* `CREATE VECTOR INDEX idx ON dataset.table(embedding) OPTIONS(index_type='IVF', distance_type='COSINE');` ([Google Cloud][9])
* Train BQML model; schedule batch `ML.PREDICT` to a table. ([Google Cloud][10])


## 48-Hour Execution Plans

### Category A (Finance) — 12-step plan (IST)

1. **D0 09:00 IST:** Confirm Rules re: external data on Kaggle **Rules/Discussions**; default to Apache-2.0/CC-BY sources. ([Kaggle][22])
2. Land sample filings (10–20 tickers) to **GCS → BQ**; normalize schema.
3. Build **feature SQL** (ratios, deltas, Altman-like, accruals).
4. Create **remote embeddings** model; embed MD\&A chunks; **CREATE VECTOR INDEX**. ([Google Cloud][7])
5. **BQML baseline** (LOGISTIC\_REG or XGBoost) for A1; evaluate. ([Google Cloud][8])
6. Add **macro joins** (BLS CPI/FED funds) for controls. ([Google Cloud][5], [FRED][6])
7. Time-aware **CV** (two slices).
8. **Calibration** and **thresholds** (distress/runway alerts).
9. **Explainability:** `enable_global_explain` and feature importance. ([GitHub][11])
10. **Vector-aware features:** top-k risk matches from MD\&A. ([Google Cloud][12])
11. **Looker Studio** dashboard + README/model card.
12. **2-min demo script**: show query → prediction → explainability → what-if.

### Category B (Tutor) — 12-step plan

1. **D0 09:00 IST:** Confirm Rules; stick to **OpenStax (CC-BY-4.0)** and **ARC (CC-BY-SA-4.0)**; avoid NC. ([Hugging Face][14])
2. Extract target chapters → chunk → load into BQ.
3. Create **remote embeddings** model; **ML.GENERATE\_EMBEDDING** for chunks. ([Google Cloud][7])
4. Build **VECTOR\_SEARCH** retrieval SQL; pin 4–6 chunks per skill. ([Google Cloud][12])
5. Create **few-shot prompts** for explain/quiz.
6. Seed 30–50 items from ARC/OpenStax; tag skill/difficulty.
7. Implement **adaptive loop** (controller service on Cloud Run hitting BQ + Vertex via remote model). ([Google Cloud][15])
8. **Safety checks**: citation-first, blocked topics list.
9. **Pre/Post tests** (10 items).
10. **State tables** in BQ (student, skill mastery, interactions).
11. Simple **web UI** (Streamlit/Cloud Run) + Looker “learning curve” chart.
12. **90s demo** recording.



## Pseudocode Templates (as requested)

**BigQuery SQL — time features & training**

```sql
-- Lag/delta example
SELECT
  ticker, filing_qtr, revenue,
  revenue - LAG(revenue) OVER (PARTITION BY ticker ORDER BY filing_qtr) AS qoq_delta,
  SAFE_DIVIDE(
    revenue - LAG(revenue) OVER (PARTITION BY ticker ORDER BY filing_qtr),
    NULLIF(LAG(revenue) OVER (PARTITION BY ticker ORDER BY filing_qtr),0)
  ) AS qoq_pct
FROM raw.financials;

-- Train BQML XGBoost classifier
CREATE OR REPLACE MODEL models.fin_cls
OPTIONS (model_type='XGBOOST_CLASSIFIER', input_label_cols=['label']) AS
SELECT * FROM feat.train_slice;
```

([Google Cloud][8])

**Adaptive Tutor Controller (minimal)**

```python
def next_step(student_id):
    s = bq.read_state(student_id)
    target = argmin_mastery(s.skills)  # lowest skill
    ctx = bq.vector_search('openstax_vectors', query=f"explain {target} to grade 10", k=4)
    explain = bq.ml_generate_text(model='vertex_gemini', context=ctx, style='concise, grade-10')
    q = pick_item(skill=target, difficulty=adaptive(s[target]))
    ans = ui.ask(explain, q)
    correct, tag = grade(ans, q)
    s = update_bkt(s, target, correct)
    if tag: s = remediate(s, tag)  # trigger targeted explanation
    bq.log_interaction(student_id, target, q.id, correct, ctx.citations)
    return s
```

([Google Cloud][12])



## Decision Matrix & Recommendation

**Scoring (1–5)**

| Idea                  | Innovation | 48h Feasibility | Judge Clarity | Data Readiness |   Rule Risk | Demo Appeal | **Total** |
| --------------------- | ---------: | --------------: | ------------: | -------------: | ----------: | ----------: | --------: |
| **A1 Distress Watch** |          4 |           **5** |             5 |              4 |           2 |       **5** |    **25** |
| A2 Cash-Runway        |          3 |           **5** |             5 |              4 |           3 |           4 |        24 |
| A3 EQ Index           |          4 |               4 |             4 |              3 |           3 |           4 |        22 |
| **B1 MechMentor**     |          4 |           **5** |             5 |          **5** |       **4** |       **5** |    **28** |
| B2 ChemCoach          |          4 |               5 |             5 |              5 | 3 (SciQ NC) |           4 |        26 |
| B3 BioCell Scout      |          4 |               5 |             5 |              5 |           4 |           4 |        27 |

**Recommendations (finalists):**

* **Category A:** **A1 — 4Q Distress Watch** (best blend of novelty, clarity, embeddings + BQML, and dashboard-ability).
* **Category B:** **B1 — MechMentor Lite** (clean licenses, crisp scope, strong adaptivity story).



## Assumptions & Unknowns Log

* **External data policy:** The Hackathon’s **Rules** page must confirm permissibility of HF/OpenStax/SEC data and pre-trained models via Vertex. If unclear, post in **Discussions** and default to open, commercial-friendly licenses (Apache-2.0, CC-BY, MIT). ([Kaggle][22])
* **Judging criteria breakdown & submission mechanics:** Confirm on **Overview**/**Discussions** (some hackathons judge via notebooks/demos vs. leaderboard). ([Kaggle][2])
* **Quota/rate limits for AI functions:** Watch **ML.GENERATE\_TEXT** concurrency errors; batch queries. ([Google Cloud Community][24])



## Appendix — Key BigQuery/Vertex Docs (for claims used)

* **BigQuery Vector Search (overview, function, indexing):** use **VECTOR\_SEARCH**, **CREATE VECTOR INDEX**. ([Google Cloud][3])
* **Embeddings from BigQuery via Vertex (remote model + ML.GENERATE\_EMBEDDING):** official guide. ([Google Cloud][7])
* **BQML training (CREATE MODEL for GLM/XGBoost/AutoML):** references and syntax. ([Google Cloud][10])
* **BigQuery AI generative functions (AI.GENERATE / ML.GENERATE\_TEXT):** docs. ([Google Cloud][26])
* **Hackathon tracks/prizes/timeline context:** Google Developers community post. (Note: always defer to Kaggle pages for official rules/timeline.) ([Google Cloud][1])



## What to Build Now (concise)

1. **Pick finalists:** A1 (Finance) + B1 (Tutor).
2. **D0 (AM IST):** Confirm Rules on Kaggle; if external data OK, proceed with **OpenStax (B)** and **EDGAR/Open BLS/FRED (A)**. ([Kaggle][22])
3. **Spin up BigQuery dataset**; create **remote embedding model**; ingest & chunk sources; **CREATE VECTOR INDEX**. ([Google Cloud][7])
4. **A1:** Feature SQL → **BQML XGBoost** → PR-AUC → Looker dashboard + explainability. ([Google Cloud][8])
5. **B1:** Vector RAG of OpenStax → adaptive loop → pre/post quiz → UI stub on Cloud Run. ([Google Cloud][12])

Ship the demo with **clean licenses, reproducible SQL, and short, visual storytelling**.

# References

- https://cloud.google.com/bigquery/public-data 
- https://www.kaggle.com/competitions/bigquery-ai-hackathon
- https://cloud.google.com/bigquery/docs/vector-search-intro
- https://huggingface.co/datasets/eloukas/edgar-corpus 
- https://console.cloud.google.com/marketplace/product/bls-public-data/cpi-unemployement?hl=en-GB
- https://fred.stlouisfed.org/series/FEDFUNDS
- https://cloud.google.com/bigquery/docs/generate-text-embedding
- https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create-glm
- https://cloud.google.com/bigquery/docs/vector-index
- https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create
- https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/quests/vertex-ai/vertex-bqml/lab_exercise.ipynb
- https://cloud.google.com/bigquery/docs/vector-search
- https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
- https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data
- https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-generate-text
- https://huggingface.co/datasets/allenai/sciq/blame/b4fbcf80d3a117943a9980f81119c1f4120cd3da/dataset_infos.json
- https://huggingface.co/datasets/JanosAudran/financial-reports-sec 
- https://huggingface.co/datasets/gtfintechlab/SubjECTive-QA
- https://huggingface.co/datasets/kensho/spgispeech
- https://huggingface.co/datasets/KrossKinetic/Bloomberg_Financial_News
- https://huggingface.co/datasets/takala/financial_phrasebank
- https://www.kaggle.com/competitions/bigquery-ai-hackathon/rules
- https://www.kaggle.com/competitions/bigquery-ai-hackathon/discussion
- https://www.googlecloudcommunity.com/gc/AI-ML/Exceeded-rate-limits-too-many-concurrent-queries-that-use-ML/m-p/864851
- https://www.kaggle.com/competitions/bigquery-ai-hackathon/discussion/598576
- https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-generate
