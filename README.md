# Can a Model Detect Its Own Failure?
## A Benchmark of Confidence, Uncertainty, Calibration, and Error Prediction

**Can a machine learning model recognize when its own prediction is likely to be wrong?**

A classifier predicts an outcome, but a reliable system should also estimate whether that prediction can be trusted.

This project investigates whether a model can anticipate its own mistakes using information available at prediction time. It benchmarks confidence-based uncertainty, ensemble disagreement, probability calibration, and learned failure detectors to understand how effectively classification errors can be identified before the ground-truth outcome is known.

Rather than evaluating models only on predictive accuracy, this benchmark examines their ability to recognize potentially unreliable predictions, quantify uncertainty, and selectively refer high-risk cases for human review.

---

## Kaggle Notebook

The original notebook **"Can a Model Detect Its Own Failure? A Benchmark of Confidence, Uncertainty, Calibration, and Error Prediction"** is available on [Kaggle](https://www.kaggle.com/code/utkarshjain76/can-a-model-detect-its-own-failure).

The notebook investigates whether a machine learning classifier can anticipate its own incorrect predictions using confidence scores, predictive uncertainty, ensemble disagreement, probability calibration, and dedicated failure-detection models.

It goes beyond conventional classification metrics by evaluating error-detection performance, selective prediction, risk–coverage trade-offs, and robustness under controlled input distribution shifts.

---

## Research Question

> Can a classifier anticipate its mistakes using information available before the ground-truth outcome is known?

The central distinction is between two tasks:

- **Classification:** What class does the model predict?
- **Failure detection:** How likely is that prediction to be incorrect?

The project evaluates whether uncertainty estimates and dedicated error-prediction models can identify incorrect predictions more effectively than conventional confidence scores.

## Project Overview

Traditional machine learning evaluation focuses on metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

However, a model can produce highly confident predictions that are still incorrect. This becomes particularly important in applications where unreliable predictions should be flagged rather than automatically accepted.

This benchmark explores:

1. Whether incorrect predictions tend to have lower confidence.
2. Whether entropy and prediction margins provide useful failure signals.
3. Whether ensemble disagreement helps identify unreliable predictions.
4. Whether probability calibration improves predictive reliability.
5. Whether dedicated machine learning models can learn to predict classification errors.
6. Whether selective prediction can reduce the error rate among accepted predictions.
7. How failure-detection performance changes under distribution shifts and input corruption.

The objective is not to construct an infallible failure detector, but to investigate the strengths and limitations of different approaches.

## Key Features

- Comparison of four classification algorithms.
- Confidence-based failure detection.
- Predictive entropy and inverse-margin analysis.
- Bootstrap ensemble uncertainty estimation.
- Probability calibration using Platt scaling, isotonic regression, and temperature scaling.
- Cross-fitted training of dedicated failure detectors.
- Error detection using AUROC and AUPRC.
- Selective prediction and human-review budget analysis.
- Risk–coverage curves and Area Under the Risk–Coverage Curve (AURC).
- Distribution-shift and input-corruption experiments.
- Paired bootstrap comparisons and ablation studies.
- Reproducible experiment outputs in CSV format.

## Datasets

The benchmark supports two binary classification datasets.

| Dataset | Description | Features |
|---|---|---|
| Breast Cancer Wisconsin Diagnostic | Historical diagnostic benchmark for classifying breast cancer observations | 569 records, 30 numerical features |
| Adult Census Income | Predicts whether annual income exceeds $50K using census attributes | Numerical and categorical features |

**Default configuration:** The notebook runs the Breast Cancer Wisconsin dataset.

The Adult Census Income experiment is optional and requires either local dataset files or internet access to retrieve the OpenML dataset.

Dataset sources:
- [UCI Breast Cancer Wisconsin Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [UCI Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult)
- [OpenML Adult Dataset](https://www.openml.org/d/1590)

These datasets are research benchmarks, not evidence of clinical suitability or equitable real-world deployment.

## Machine Learning Models

Four classification algorithms are evaluated:

| Model | Purpose |
|---|---|
| Logistic Regression | Linear probabilistic baseline |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Boosted tree-based classification |
| Multi-Layer Perceptron (MLP) | Neural network-based classification |

The base classifier is selected using development-set log loss rather than the final test set.

## Methodology

The benchmark follows a structured experimental pipeline.

```text
Dataset Loading
      |
      v
Exploratory Data Analysis
      |
      v
Leakage-Safe Preprocessing
      |
      v
Train / Development / Test Split
      |
      v
Base Classifier Training
      |
      v
Development-Based Model Selection
      |
      v
Cross-Fitted Predictions
      |
      v
Failure-Prediction Dataset
      |
      v
Dedicated Failure Detectors
      |
      v
Calibration and Uncertainty Analysis
      |
      v
Selective Prediction
      |
      v
Distribution Shift Experiments
      |
      v
Statistical Analysis and Ablations
      |
      v
Consolidated Results
```

### 1. Leakage-Safe Data Splitting

The dataset is divided into three partitions:

| Partition | Percentage | Purpose |
|---|---:|---|
| Base-training | 60% | Train the base classifier and generate out-of-fold predictions |
| Development | 20% | Model selection, calibration, detector and policy selection |
| Test | 20% | Independent final evaluation |

Three-fold cross-validation within the base-training partition generates out-of-fold predictions for training the failure detectors.

This prevents a training observation from being used to train the classifier that generates its own meta-training prediction.

The final test partition is kept separate from model and policy selection.

### 2. Confidence-Based Failure Detection

The simplest failure-risk score is:

`Confidence Risk = 1 - max(P(y | x))`

A prediction with a lower maximum class probability receives a higher estimated risk.

The benchmark compares this baseline with other uncertainty measures.

### 3. Predictive Uncertainty

Three principal uncertainty measures are investigated.

**Predictive Entropy**

Measures uncertainty in the predicted class probabilities.

`H(p) = -Σ p(k) log(p(k)) / log(K)`

**Inverse Prediction Margin**

Measures the difference between the highest and second-highest predicted class probabilities.

`Risk = 1 - (p(1) - p(2))`

**Ensemble Disagreement**

Multiple independently fitted Random Forest models generate predictions. Their disagreement is used to derive:

- Ensemble entropy
- Ensemble variance
- Ensemble mutual-information-style proxy
- Ensemble vote disagreement

The ensemble information-gap measure is treated as a heuristic, not as a guaranteed Bayesian estimate of epistemic uncertainty.

**Important mathematical observation:** For binary classification, confidence risk, normalized entropy, and inverse margin have equivalent ranking behaviour, apart from ties. They should not be interpreted as three independent sources of information.

### 4. Dedicated Failure-Prediction Models

Instead of relying exclusively on confidence scores, the project trains separate classifiers to predict whether the base classifier will make an error.

The failure label is defined as:

`Error = 1 if predicted_class != actual_class else 0`

The following models are investigated as failure detectors:

- Logistic Regression
- Random Forest
- Gradient Boosting

The detectors learn from cross-fitted predictions and their corresponding error labels.

The objective is to estimate:

`P(Base Model Prediction Is Incorrect | Available Prediction Features)`

The final detector is evaluated on independent development and test predictions.

### 5. Probability Calibration

A model can produce accurate class predictions while generating poorly calibrated probabilities.

The benchmark compares:

- Raw model probabilities
- Platt scaling
- Isotonic regression
- Temperature scaling

Calibration quality is examined using:

- Log Loss
- Brier Score
- Expected Calibration Error (ECE)
- Reliability diagrams

The notebook distinguishes probability calibration from error-ranking performance. Better-calibrated probabilities do not necessarily imply better failure detection.

### 6. Failure Detection Evaluation

Failure detectors are evaluated by treating incorrect predictions as the positive class.

Primary metrics:

| Metric | Purpose |
|---|---|
| Error AUROC | Measures the ability to rank incorrect predictions above correct ones |
| Error AUPRC | Measures precision–recall performance for identifying errors |
| Precision | Fraction of flagged predictions that are incorrect |
| Recall | Fraction of actual errors successfully identified |

AUPRC must be interpreted relative to the observed error prevalence. A seemingly reasonable AUROC does not necessarily mean that a detector will identify errors effectively in practice.

### 7. Selective Prediction and Human Review

A practical failure detector should help determine which predictions can be accepted automatically and which should be referred for review.

The benchmark ranks predictions according to estimated risk and investigates fixed review budgets:

- 5%
- 10%
- 20%
- 30%

It evaluates:

- Selective risk among accepted predictions
- Coverage of automatically accepted predictions
- Errors missed by the review policy
- Correct predictions unnecessarily referred for review

The Area Under the Risk–Coverage Curve (AURC) is also calculated.

This provides a more practical perspective on reliability than evaluating error-detection AUROC alone.

### 8. Distribution Shift and Robustness

The benchmark investigates how failure-detection methods behave when input data changes.

Experiments include:

| Experiment | Description |
|---|---|
| Measurement Noise | Adds Gaussian noise to numerical features |
| Additional Missingness | Introduces randomly missing numerical values |
| Numeric Drift | Applies systematic shifts to selected numerical features |
| Subgroup Analysis | Examines a predefined demographic or subgroup slice when available |

The original trained pipeline is used to evaluate corrupted inputs without refitting the detector on corrupted test data.

Labels are held fixed in the numeric-corruption experiments. These are sensitivity analyses, not proof that every perturbed observation remains a valid real-world example.

### 9. Statistical Significance and Ablation Studies

A paired bootstrap comparison estimates the difference in detector error-AUROC on common test examples.

The notebook also investigates the contribution of different meta-features:

1. Confidence alone.
2. Confidence, entropy, and inverse margin.
3. Additional predicted-class information.
4. Additional raw numerical input features.
5. Additional ensemble disagreement features.

The final test set is used for estimation, not for selecting the winning ablation.

The bootstrap analysis is exploratory and does not correct for repeated comparisons.

## Experimental Design and Reproducibility

The notebook uses:

- Fixed random seed: `42`
- Stratified train/development/test splitting
- Three-fold out-of-fold prediction generation
- Leakage-safe preprocessing pipelines
- Development-based model selection
- Independent final test evaluation
- Logged Python and library versions

### Fast Mode and Full Mode

The notebook supports two execution configurations.

| Configuration | Description |
|---|---|
| Fast Mode | Reduced model complexity for quick execution and smoke testing |
| Full Mode | More estimators and iterations, with additional ensemble experiments |

Set the environment variable:

```python
FAST_MODE = True
```

for faster experimentation, or:

```python
FAST_MODE = False
```

for the more extensive benchmark.

The notebook defaults to full mode unless the `FAST_MODE` environment variable is set to `1`.

## Generated Results and Artifacts

After execution, the notebook exports the following experiment artifacts.

| File | Description |
|---|---|
| `prediction_level_results.csv` | Prediction-level probabilities, confidence, errors, and risk estimates |
| `base_model_performance.csv` | Base classifier evaluation results |
| `calibration_results.csv` | Calibration performance across different methods |
| `failure_detection_leaderboard.csv` | Failure-detector evaluation metrics |
| `risk_coverage_results.csv` | Selective prediction and review-budget results |
| `distribution_shift_comparison.csv` | Performance under original and corrupted inputs |
| `meta_ablation_results.csv` | Failure-detector feature ablation results |
| `runtime_and_selection.csv` | Runtime measurements and selected methods |
| `high_confidence_by_model.csv` | Analysis of high-confidence incorrect predictions |
| `reproducibility.json` | Experiment configuration, random seed, dataset sources, and library versions |

On Kaggle, outputs are saved under:

`/kaggle/working/`

On Google Colab, outputs are saved under:

`/content/failure_benchmark_outputs/`

The notebook also contains an optional cell to package the generated CSV files into a ZIP archive.

## Key Research Questions

The notebook investigates eight central questions:

1. How frequently does the classifier make incorrect predictions with high confidence?
2. How effective is maximum confidence at detecting classification errors?
3. Can ensemble uncertainty improve failure detection?
4. Does probability calibration improve predictive probability quality?
5. Can learned failure detectors outperform confidence-based baselines?
6. How much can selective prediction reduce the error rate at a fixed review budget?
7. How does distribution shift affect classification and failure-detection performance?
8. What is the computational overhead of uncertainty estimation and learned failure detection?

**Results are generated from the executed experiments.** No aggregate performance figures are assumed or hard-coded in the README.

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy
- Jupyter Notebook
- Kaggle Notebooks
- Google Colab

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/JainUtkarsh18/Can-a-Model-Detect-Its-Own-Failure.git
cd Can-a-Model-Detect-Its-Own-Failure
```

Replace the repository URL if your GitHub repository uses a different name.

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib scipy jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook
```

Open:

`Can_a_Model_Detect_Its_Own_Failure.ipynb`

### 4. Run the experiments

Run all cells sequentially.

The default experiment uses the Breast Cancer Wisconsin dataset and does not require an external dataset download.

To run the optional Adult Census Income experiment, provide `adult.data` and `adult.test` locally or enable internet access for the OpenML download.

For the full benchmark, use:

```bash
FAST_MODE=0
```

For a quicker run:

```bash
FAST_MODE=1
```

## Repository Structure

```text
Can-a-Model-Detect-Its-Own-Failure/
│
├── Can_a_Model_Detect_Its_Own_Failure.ipynb
│
├── outputs/
│   ├── prediction_level_results.csv
│   ├── base_model_performance.csv
│   ├── calibration_results.csv
│   ├── failure_detection_leaderboard.csv
│   ├── risk_coverage_results.csv
│   ├── distribution_shift_comparison.csv
│   ├── meta_ablation_results.csv
│   ├── runtime_and_selection.csv
│   ├── high_confidence_by_model.csv
│   └── reproducibility.json
│
└── README.md
```

The output files are generated after executing the notebook; they do not need to be committed to the repository.

## Limitations

Several limitations should be considered when interpreting the results:

- Only two binary classification datasets are currently supported.
- The Breast Cancer Wisconsin dataset is relatively small.
- Adult Census Income represents historical U.S. census data.
- Artificial input corruptions do not fully reproduce real-world distribution shifts.
- The OOF meta-training prediction distribution may differ from predictions generated by a model fitted on the complete training partition.
- Detector selection can overfit the development data through repeated comparisons.
- Bootstrap comparisons are exploratory and do not correct for multiple testing.
- Human review is not experimentally validated; referred errors are counted under an idealized review assumption.
- No experiment establishes clinical suitability, fairness, or safe deployment.

A larger study should incorporate more datasets, repeated outer resampling, genuine external distribution shifts, decision-theoretic review costs, and prospective monitoring.

## Future Work

Potential extensions include:

- More diverse tabular and real-world datasets.
- Multiclass failure detection.
- Deep learning-based failure detectors.
- Out-of-distribution detection.
- Conformal prediction and uncertainty estimation.
- Subgroup-specific calibration and reliability analysis.
- Cost-sensitive selective prediction.
- More rigorous statistical comparisons.
- Prospective evaluation of human-in-the-loop review systems.

## References

1. Guo et al. (2017). [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). ICML.
2. Lakshminarayanan et al. (2017). [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474). NeurIPS.
3. Geifman and El-Yaniv (2017). [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500). NeurIPS.
4. Hendrycks and Gimpel (2017). [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks](https://arxiv.org/abs/1610.02136). ICLR.
5. Naeini, Cooper, and Hauskrecht (2015). [Obtaining Well Calibrated Probabilities Using Bayesian Binning](https://doi.org/10.1609/aaai.v29i1.9602). AAAI.

## Conclusion

A machine learning model should not only predict an answer; it should also provide useful information about the reliability of that prediction.

This benchmark moves beyond conventional classification metrics to investigate whether confidence, uncertainty, ensemble disagreement, calibration, and dedicated failure detectors can help identify incorrect predictions.

The broader objective is to make machine learning systems more reliable by understanding not just when they are correct, but also when they are likely to fail.

> **Predicting correctly is important. Knowing when a prediction might be wrong is equally important.**

---

**Contributions, suggestions, and research collaborations are welcome.**

If you find this project interesting, consider starring the repository!
