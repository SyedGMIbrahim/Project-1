# Symptom Schema Limitations & Dataset Coverage Gaps

This document tracks identified gaps between real-world clinical vocabulary (extracted accurately by the LLM) and the 230-column feature schema in `Diseases_and_Symptoms_dataset.csv`. Because the pipeline is strictly grounded to the dataset schema, missing columns will result in dropped symptoms and potentially degraded prediction accuracy for the affected diseases.

## Architecture Facts
- **Disease Classes**: Corrected: the dataset contains 100 disease classes, not 77 as originally documented.
- **Meta-Learner**: The stacking meta-learner remains an MLPClassifier to match the original architecture, but it is restricted to a shallow network (`hidden_layer_sizes=(20,)`, `max_iter=100`, `early_stopping=True`) to make training computationally tractable on CPU.

## Future Work: Mitigating Feature Sparsity

### Failed Experiment: Dropout Augmentation
We attempted to mitigate the sparse-vector overconfidence issue (where the model becomes overly confident when only a few symptoms are present, e.g., predicting 'hyperemesis gravidarum' at 99.71% confidence or 'diaper rash' at 98.39% confidence with only 3-4 symptoms) by applying random sparse dropout (30-85%) during training. 

**Methodology:**
1. We generated 1 augmented copy per row in the training set (yielding 195,000 total rows).
2. We used `GroupKFold` splits based on the parent row ID to prevent augmented duplicates from leaking across the StackingClassifier's meta-learner CV folds and the final train/holdout split.
3. We retrained the full StackingClassifier (RandomForest, ExtraTrees, HistGradientBoosting) with an MLP meta-learner.

**Results:**
The experiment resulted in a broad regression across all metrics and real-world test cases:
*   **Original Holdout Accuracy:** Dropped from ~94% to **88.73%**.
*   **Augmented Holdout Accuracy:** Reached only **75.22%**.
*   **Real-world Tests:** Failed all 5 benchmark cases (Cold, Migraine, Diabetes, Chickenpox, UTI). For example, UTI mapped to `['frequent urination', 'involuntary urination', 'lower abdominal pain', 'painful urination']` but predicted 'temporary or benign blood in urine' (65.49%). The overconfidence issue persisted (Chickenpox predicted as 'diaper rash' at 98.39%).

**Conclusion:**
Naïve random feature dropout destroys critical co-occurrence signals the tree-based estimators rely on, leading to severe underfitting on common symptom presentations. 

### Future Work Strategies
To properly address the sparsity and overconfidence problems without degrading overall accuracy, future iterations should explore:
1. **Domain-Aware Augmentation:** Instead of uniform random dropout, use symptom-frequency priors to drop rare symptoms while preserving core hallmark symptoms of a disease.
2. **Confidence Calibration:** Apply isotonic regression or Platt scaling on the meta-learner's output to better calibrate the probabilities for sparse inputs, rather than altering the training data distribution.
3. **Generative Oversampling:** Use SMOTE or generative approaches to synthesize realistic sparse presentations rather than destructively dropping features from dense records.

## Known Gaps

### Migraine (Neurological / Sensory)
- **Missing Columns**: There are no columns to represent sensory aura or light/sound sensitivity.
  - *Dropped Symptoms*: `photophobia` (light sensitivity), `phonophobia` (sound sensitivity), `visual aura` (e.g., flickering zigzag lines).
- **Impact**: The model is forced to classify migraines using only generic pain/nausea symptoms, which overlap heavily with other conditions (e.g., hyperemesis gravidarum), leading to severe misclassification and low confidence.

### Diabetes (Metabolic / Healing)
- **Missing Columns**: There are no columns to represent increased thirst (polydipsia) or delayed wound healing.
  - *Dropped Symptoms*: `excessive thirst` (or polydipsia, dehydration), `slow-healing wounds` (or unhealed sores, delayed healing).
- **Asymmetric Coverage Flaw (Weight)**: 
  - *Dropped Symptom*: `unintentional weight loss`
  - The dataset has a `weight gain` column but **completely lacks a `weight loss` column**. This is a significant data-completeness flaw, as weight loss is a classic diagnostic criterion for many conditions (diabetes, cancer, hyperthyroidism) whereas weight gain covers a different clinical subset.
- **Impact**: For a disease defined by polyuria, polydipsia, polyphagia, and weight loss, only `frequent urination` successfully maps to the dataset. The resulting sparse vector (1-2 generic symptoms) causes the classifier to misfire (e.g., predicting BPH instead of diabetes).

### Remediation Strategy
To maintain clinical accuracy, we have implemented a strict **"Unmapped" logic**:
1. **Semantic Similarity Floor**: The embedding-based matching (using `all-MiniLM-L6-v2`) now enforces a strict `0.65` cosine similarity floor.
2. **Explicit Unmapped Logging**: Any valid symptom described by the user that does not cross this threshold for any of the 230 columns is explicitly dropped from the prediction array and logged as: 
   > `Symptoms described but not supported by current symptom schema: [symptom]`

This ensures that the model only predicts based on grounded, supported data while accurately reflecting the limitations of our current feature set.

> [!WARNING]
> This limitation is confirmed to be a real gap in the underlying dataset. Until the dataset is expanded and the model retrained, diagnoses for conditions heavily reliant on sensory sensitivities (like specific migraine variants) may be under-represented or under-confident.
