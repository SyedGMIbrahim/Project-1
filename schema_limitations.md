# Symptom Schema Limitations & Dataset Coverage Gaps

This document tracks identified gaps between real-world clinical vocabulary and the 230-column feature schema in `Diseases_and_Symptoms_dataset.csv`, as well as the evolution of the extraction pipeline and remediation experiments. Because the pipeline is strictly grounded to the dataset schema, missing columns result in dropped symptoms and degraded prediction accuracy.

## 1. Architecture Facts
- **Disease Classes**: Corrected: the dataset contains 100 disease classes, not 77 as originally documented.
- **Meta-Learner**: The stacking meta-learner remains an `MLPClassifier` to match the original architecture, but it is restricted to a shallow network (`hidden_layer_sizes=(20,)`, `max_iter=100`, `early_stopping=True`) to make training computationally tractable on CPU.

## 2. Extraction Pipeline Evolution
The process of mapping free-text user input to the strict 230-column dataset schema went through several iterations:
1. **Hallucination (Initial state)**: The LLM was initially used to extract and map symptoms directly to the schema. However, it hallucinated columns or forced incorrect mappings (e.g., mapping unrelated terms to generic categories).
2. **Over-matching**: We switched to extracting free-text symptoms first and mapping them using difflib. This led to over-matching, where short or vaguely similar strings mapped incorrectly.
3. **Semantic Mapping with Synonym Enrichment (Current state)**: We implemented a hybrid approach:
   - Difflib fast-path for exact or near-exact matches.
   - Semantic similarity matching using `all-MiniLM-L6-v2` embeddings with a strict `0.65` cosine similarity floor.
   - A curated synonym dictionary (enriching generic columns with common clinical terms).
   - Explicit logging of "Unmapped" symptoms that fail to cross the similarity threshold, ensuring the model only predicts based on grounded data.

> [!NOTE]
> This extraction component is now considered **resolved and stable**. A working example is the UTI test case, which successfully extracted and mapped all critical symptoms (`['frequent urination', 'lower abdominal pain', 'painful urination']`) with a clean 73% confidence.

## 3. Confirmed Dataset Gaps
Even with robust semantic mapping, several critical clinical features are entirely missing from the underlying dataset:

### Migraine (Neurological / Sensory)
- **Missing Columns**: No columns represent sensory aura, light sensitivity (photophobia), or sound sensitivity (phonophobia).
- **Impact**: The model classifies migraines using only generic pain/nausea symptoms, which overlap heavily with other conditions (e.g., hyperemesis gravidarum), leading to severe misclassification and low confidence.

### Diabetes (Metabolic / Healing)
- **Missing Columns**: No columns represent increased thirst (polydipsia) or delayed wound healing.
- **Asymmetric Coverage Flaw (Weight)**: The dataset has a `weight gain` column but **completely lacks a `weight loss` column**. Weight loss is a classic diagnostic criterion for many conditions (diabetes, cancer, hyperthyroidism).
- **Impact**: For a disease defined by polyuria, polydipsia, polyphagia, and weight loss, only `frequent urination` successfully maps to the dataset. The resulting sparse vector (1-2 generic symptoms) causes the classifier to misfire (e.g., predicting BPH instead of diabetes).

## 4. Failed Remediation Attempts
To address the "sparse-vector overconfidence" issue (where the model is >90% confident on entirely incorrect predictions for sparse inputs like 2-3 generic symptoms), we ran two substantive ML-side experiments. Both were documented negative results.

### Failed Experiment 1: Dropout Augmentation
We attempted to force the model to learn from sparse presentations by applying random sparse dropout (30-85%) during training.
- **Methodology**: Generated 1 augmented copy per row in the training set (yielding 195,000 total rows). Used `GroupKFold` splits to prevent augmented duplicates from leaking across the StackingClassifier's meta-learner CV folds. Retrained the full ensemble.
- **Results**: Broad regression across all metrics. Holdout accuracy dropped from ~94% to **88.73%**. Augmented holdout accuracy reached only **75.22%**. Failed all 5 real-world benchmark cases. Overconfidence persisted.
- **Root Cause**: Naïve random feature dropout destroys critical co-occurrence signals the tree-based estimators rely on, leading to severe underfitting on common symptom presentations.

### Failed Experiment 2: Confidence Calibration
We attempted to calibrate the overconfident probabilities using Platt scaling on the meta-learner's output, rather than altering the training data.
- **Methodology**: Applied `CalibratedClassifierCV` (method="sigmoid", cv="prefit") directly on the existing production model, fitted on the original 20% holdout test set. 
- **Results**: Overall holdout accuracy remained essentially unchanged (88.17% vs 88.23%). However, sparse-vector cases remained >90% confident and wrong. In one sparse case (diabetes), the confidence actually *increased* (from 78.30% to 90.52%).
- **Root Cause**: The holdout set used to fit the calibration curve is itself dense and well-populated. The sigmoid function learned that "high confidence = trustworthy" from a distribution that never contains the sparse, low-information vectors that cause the overconfidence problem. Calibration cannot correct for a failure mode it never observes in its fitting data.

## 5. Future Work
All ML-side attempts point to the same underlying cause: the training data itself lacks realistic sparse/partial symptom presentations and lacks certain clinical features entirely. To properly address these issues beyond the current project scope, future iterations should explore:

1. **Domain-Aware Augmentation:** Instead of uniform random dropout, use symptom-frequency priors to drop rare symptoms while preserving core hallmark symptoms of a disease.
2. **Generative Oversampling:** Use SMOTE or generative LLM approaches to synthesize realistic sparse/partial presentations, rather than destructively dropping features from dense records.
3. **Sourcing a Richer Dataset:** The ultimate fix is migrating to a dataset that natively includes critical missing features (weight loss, sensory auras, polydipsia) and naturally contains realistic variance in symptom sparsity.
4. **Feature Importance Review:** Revisit known quirks in feature weighting, such as the common-cold-vs-allergies "sneezing" quirk, where the model heavily overweighs highly generic symptoms.

---

## 6. Deployment Gap & Extraction Pipeline Fixes — 2026-09-01

### 6.1 The Bug: Verified Fix Not Live in the Running Application

After the context-preserving LLM prompt fix was written to `main.py` and confirmed via a standalone test script, the live application still served stale in-memory code. The root cause was that the uvicorn process had not been restarted since the file was edited, and the test script imported `extract_symptoms_with_llm` directly (picking up the fresh code from disk) while the running `/api/extract-symptoms` endpoint had a **completely separate, duplicated copy** of the extraction logic that had never received the fix.

The gout and conjunctivitis inputs through the live `/api/predict` endpoint continued to produce the pre-fix hallucinated symptoms (stomach bloating for gout, vaginal discharge for conjunctivitis), exposing the gap.

**Lesson documented here explicitly:** A verified fix to `main.py` is only confirmed live when:
1. The running uvicorn process has been restarted (or `--reload` has picked up the change and logged "Application startup complete")
2. The fix has been tested through the **actual HTTP endpoint** (`/api/extract-symptoms`, `/api/diagnose`) via `curl` or the frontend — not through a test script that imports Python functions directly and bypasses the request path entirely.

### 6.2 Three Root Causes Found

| # | Root Cause | Where |
|---|---|---|
| 1 | **Duplicate endpoint logic** — `/api/extract-symptoms` had its own inline copy of the extraction code, independent of `extract_symptoms_with_llm`. The shared function received the context-preservation fix; the endpoint copy did not. | `main.py` L547–582 (old) |
| 2 | **No temperature setting on Ollama call** — Llama 3 8B defaulted to temperature ~0.8, causing extraction completeness to vary dramatically across calls. One run extracted 2 symptoms, the next extracted 4, with no prompt change. | `extract_symptoms_with_llm` |
| 3 | **Cross-anatomy mapping hallucination above the similarity floor** — `"thick yellowish discharge"` (extracted correctly by the LLM for an eye patient) mapped to `vaginal discharge` (cosine score: 0.816) rather than `white discharge from eye` (score: 0.648) because the phrase lacked the word "eye". `vaginal discharge` had a stronger generic embedding for the word "discharge" and scored above the 0.65 floor, so the guard never triggered. | `map_extracted_to_features` |

### 6.3 Fixes Applied

1. **Consolidated to single source of truth** — deleted the inline extraction block from the `/api/extract-symptoms` route handler; it now calls `extract_symptoms_with_llm` exclusively. One function, one prompt, one place to maintain.

2. **Prompt rewritten for exhaustive extraction** — the new `SYMPTOM_EXTRACTION_PROMPT_TEMPLATE` has numbered rules explicitly stating "do not stop after the first one", "treat each distinct complaint as a separate item", and cites an example with 5 distinct entries. The old prompt said "extract all... do not summarize" which Llama 3 8B routinely ignored for longer inputs.

3. **`temperature=0.0` + `num_predict=512`** — Ollama call now has deterministic sampling. 4/4 identical runs confirmed on both gout and conjunctivitis after this change.

4. **Anatomical conflict guard in `map_extracted_to_features`** — when the extracted phrase contains an eye-context qualifier (`eye`, `eyelash`, `gritty`, `ocular`, etc.) but the best semantic match is a genital/non-eye feature (`vaginal discharge`, `penile discharge`), the match is rejected and the next candidate is tried. This is a principled structural fix, not a patch for one synonym.

5. **Expanded `CURATED_SYNONYMS`** — added explicit synonym lists for:
   - `white discharge from eye` → covers "thick yellowish discharge", "crusty eyelashes", "crusty eye discharge", "discharge from eye"
   - `foreign body sensation in eye` → covers "gritty eye", "grittiness in eye", "sand in eye", "eye feels gritty"
   - `lacrimation` → covers "watery eye", "watering eye", "watering right eye"
   - `eye redness` → covers "red eye", "red right eye", "bloodshot eye"
   - `itchiness of eye` → covers "itchy eye", "itchy right eye"
   - `foot or toe swelling` → covers "swollen big toe", "swelling in big toe"
   - `foot or toe pain` → covers "big toe pain", "pain in big toe", "sudden severe pain in big toe"

6. **Startup prompt logging** — `lifespan` now logs the full `SYMPTOM_EXTRACTION_PROMPT_TEMPLATE` text at server startup, making it immediately visible in server logs whether the live process is running the current or a stale version of the prompt.

### 6.4 8-Case Regression Suite — Final Post-Fix Results (2026-09-01)

All 8 cases run against the live `/api/extract-symptoms` + `/api/diagnose` endpoints with `temperature=0.0`, rewritten prompt, paresthesia synonyms, and 0.65 abstention threshold active.

| Case | Extracted Symptoms | Prediction | Confidence | Status |
|---|---|---|---|---|
| **Cold** | fever, ache all over, sore throat, coryza, cough | otitis media | 69% | ⚠️ Known schema misclassification — no regression |
| **Migraine** | headache, nausea | hyperemesis gravidarum | 91% | ⚠️ Known gap (photophobia/phonophobia absent) — no regression |
| **Diabetes** | **paresthesia**, fatigue, diminished vision, frequent urination | **Non-specific (low confidence)** | 64.8% | ✅ **Mislabeling fixed** — was mapping "tingling in feet" → `loss of sensation`; now correctly maps to `paresthesia`. Abstention fires correctly (64.8% < 65% threshold). Top internal candidate is MS (schema gap: no polydipsia/thirst column) |
| **Chickenpox** | fever, headache, skin rash, **itching of skin** | strep throat | 92% | ⚠️ Slight improvement — `itching of skin` replaced prior `allergic reaction` mapping. Prediction unchanged (vesicular rash absent from schema) |
| **UTI** | painful urination, low urine output, frequent urination, unusual color/odor urine, lower abdominal pain | **Non-specific (low confidence)** | 50% | ⚠️ Correctly abstaining — UTI surfaces in `probabilities` dict (50%). Prediction genuinely ambiguous at this feature vector |
| **Gout** | foot or toe pain, foot or toe swelling | **Non-specific (low confidence)** | 53% | ⚠️ Correctly abstaining — Gout surfaces in `probabilities` dict (53%). Schema gap: joint redness/heat absent |
| **Conjunctivitis** | itchiness of eye, eye redness, white discharge from eye, lacrimation | **conjunctivitis** | **99%** | ✅ Correct, no hallucinations, stable |
| **Hypothyroidism** | chills, weight gain, sleepiness, skin dryness | obstructive sleep apnea | 99% | ⚠️ Known schema gap — no regression |

### 6.5 Ongoing Schema Gap: Gritty-Eye / Foreign Body Sensation

The fifth conjunctivitis symptom (gritty/foreign-body sensation, "feels like sand in the eye") is consistently extracted correctly by the LLM but falls below the 0.65 cosine similarity floor even with synonyms. `foreign body sensation in eye` embedding doesn't cluster tightly with "grittiness" — scores land around 0.59–0.63. Logged as:

```
WARNING: Symptoms described but not supported by current symptom schema: gritty right eye
```

This is correct behavior: the symptom is real, the extraction is correct, the schema lacks a mappable column. Prediction is unaffected — 4 mapped eye symptoms produce 99% conjunctivitis confidence.

### 6.6 Two Additional Mapper Bugs Found and Fixed (post-regression-review)

Both bugs were found by refusing to accept the initial regression report at face value and investigating flagged cases individually.

#### Bug 4 — `"tingling in feet"` mapped to `loss of sensation` instead of `paresthesia`

**Source:** User flagged that the diabetes extraction returned `"loss of sensation"` which is not in the source text. Investigation showed:
- Raw Ollama output (verified by calling the API directly, bypassing the endpoint): `["extreme thirst", "frequent urination", "fatigue", "blurred vision", "tingling in feet"]` — correct, grounded in source text.
- `map_extracted_to_features` computed cosine scores: `loss of sensation` = **0.767**, `paresthesia` = 0.744. The correct column lost by 0.023.
- These are clinically distinct: `loss of sensation` = hypoesthesia (negative); `paresthesia` = tingling/pins-and-needles (positive/paresthetic).

**Fix:** Added `paresthesia` to `CURATED_SYNONYMS` with 11 explicit synonyms including "tingling in feet", "tingling in toes", "pins and needles", "prickling sensation". These are precomputed into the embedding index at startup. Result: `paresthesia` is now selected correctly.

**Residual:** With `paresthesia` correctly mapped (instead of the MS-steering `loss of sensation`), the top-1 prediction shifts but the 0.65 abstention threshold fires (64.8% < 65%), suppressing a confident wrong answer. The correct disease (diabetes) still doesn't appear in top-3 because the schema lacks a polydipsia/thirst column — a documented gap, not a mapper error.

#### Bug 5 — Abstention threshold set at 0.60, not 0.65, and not logged

The abstention feature was present but the threshold was 0.60, causing the `"Non-specific symptoms (low confidence)"` label to fire only at very low confidence. The regression table incorrectly marked UTI (50%) and Gout (53%) as ✅ when both were already abstaining — they were correct behaviors labelled with wrong symbols.

**Fix:** Raised `ABSTENTION_THRESHOLD` to 0.65, added explicit `LOGGER.info` log line on every abstention event, and kept the top-candidate disease in `probabilities` dict so the frontend can display "UTI suspected (low confidence)" rather than a blank non-specific label.

**Effect on 8-case suite at 0.65 threshold:** 5 cases answer confidently — Cold (69%), Migraine (91%), Chickenpox (92%), Conjunctivitis (99%), Hypothyroidism (99%). 3 cases correctly abstain — Diabetes (64.8%), UTI (50%), Gout (53%).

### 6.7 Structural Observation: Near-Tie Vulnerability in Pure Cosine-Similarity Ranking

The diabetes paresthesia/loss-of-sensation mislabeling (margin: 0.023) is the **third instance this session** where a near-tie between two clinically distinct schema columns produced a wrong mapping:

1. **Conjunctivitis** — `"thick yellowish discharge"` → `vaginal discharge` (0.816) over `white discharge from eye` (0.648). Margin: 0.168. Fixed by anatomical conflict guard + synonym expansion.
2. **Gout** — earlier-session mapping produced `stomach bloating` over `foot or toe pain` for a gout presentation. Fixed by synonym expansion.
3. **Diabetes** — `"tingling in feet"` → `loss of sensation` (0.767) over `paresthesia` (0.744). Margin: 0.023. Fixed by synonym expansion.

Each instance was found only because a user or downstream output flagged an anatomically impossible or clinically inconsistent result. None were caught by automated unit tests, because the tests verified pipeline plumbing, not clinical correctness of the mapping.

**Structural framing for the report:** Pure cosine-similarity ranking selects the globally highest-scoring candidate without any awareness of clinical specificity, anatomical locality, or the discriminative importance of near-ties. A 0.023-point margin is not a meaningful confidence signal — at that granularity, embedding noise and tokenization artefacts are as influential as genuine semantic distance. The current synonym patches and anatomical conflict guard mitigate specific known instances but do not address the underlying vulnerability.

**Recommended long-term fixes (future work):**

5. **Minimum-margin requirement:** Reject a match unless the top-1 score exceeds top-2 by a minimum margin (e.g., Δ ≥ 0.05). Near-ties would fall to the `Unmapped` path and be logged, surfacing ambiguous cases for human review rather than silently picking the wrong column.
6. **Clinical ontology / ICD-mapping layer:** Map extracted phrases to ICD-10 or SNOMED-CT codes first, then resolve codes to schema columns. Two concepts that are semantically close in embedding space (tingling ≈ loss of sensation) are clearly distinct ICD codes (R20.2 paraesthesia vs R20.0 anaesthesia of skin), eliminating this class of error entirely without per-symptom patching.
7. **Discriminative re-ranking:** After cosine ranking, apply a lightweight re-ranker trained on (phrase, schema-column) pairs labelled by a clinical ontology. This is more practical than full ICD integration and would generalize across future near-ties not yet observed.
