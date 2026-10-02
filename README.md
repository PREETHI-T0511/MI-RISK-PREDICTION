# Myocardial Infarction (Heart Attack) Risk Prediction

Mini-project for **UE24CS352A – Machine Learning** (reproduce and extend).

This project reproduces the experiments in the reference paper *"Predicting myocardial infarction risk using
supervised and unsupervised machine learning methods"* (Shi, Bechler, Ho) and extends it with one additional model,
**XGBoost**, evaluated on the same data split and metrics.

## Team

- Adarsha V
- Preethi T

## Dataset

[Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)
(Kaggle): 303 patients, 13 clinical features (age, sex, chest pain type, resting blood pressure, cholesterol,
fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, ST slope, number of
major vessels, thallium test) and a binary target (`output`: 1 = high MI risk, 0 = low MI risk).
The file is stored at `data/heart.csv`.

## Project structure

```
.
├── data/
│   └── heart.csv
├── notebooks/
│   ├── part1_reproduction.ipynb   # Part 1: reproduce the paper
│   ├── part2_xgboost.ipynb        # Part 2: XGBoost extension
│   └── results/                   # saved tables (CSV) and figures (PNG)
├── requirements.txt
└── README.md
```

## Setup and how to run

Requires Python 3.10+.

```bash
git clone https://github.com/PREETHI-T0511/MI-RISK-PREDICTION.git
cd MI-RISK-PREDICTION

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook
```

Open the notebooks inside `notebooks/` and run them **in this order** (Run All):

1. `part1_reproduction.ipynb` – saves its results to `notebooks/results/`.
2. `part2_xgboost.ipynb` – uses the same split and loads the Part 1 results for comparison.

Both notebooks use `RANDOM_STATE = 42` and read the data from `../data/heart.csv`, so run them from the
`notebooks/` folder (this is the default when opened in Jupyter or VS Code).

## Part 1 – Reproduction

- Exploratory analysis (correlation matrix, target distribution).
- Unsupervised: PCA, K-means, DBSCAN (evaluated with Adjusted Rand Index).
- Supervised: logistic regression, gradient boosting, SVM (linear / RBF / polynomial), elastic net, bagging,
  random forest, neural network.
- Stratified 60/20/20 train/validation/test split; hyperparameters tuned with 3x repeated 10-fold cross-validation;
  models compared on AUROC, precision, recall, F1 and accuracy.
- Reported vs. reproduced comparison: `notebooks/results/reported_vs_reproduced.csv`.

## Part 2 – Extension: XGBoost

XGBoost is not used in the reference paper. It is a regularised, optimised gradient-boosting implementation, so it
is a fair comparison against the paper's gradient boosting model and its best model (logistic regression).
It was tuned with the same cross-validation scheme and evaluated on the same validation and test sets.

### Test-set results

| Model | Accuracy | AUROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| SVM (RBF) | 0.87 | 0.93 | 0.82 | 0.97 | 0.89 |
| Bagging | 0.84 | 0.93 | 0.81 | 0.91 | 0.86 |
| **XGBoost (extension)** | 0.82 | 0.93 | 0.79 | 0.91 | 0.85 |
| Random forest | 0.82 | 0.93 | 0.79 | 0.91 | 0.85 |
| Logistic regression | 0.82 | 0.93 | 0.81 | 0.88 | 0.84 |
| Gradient boosting | 0.80 | 0.90 | 0.80 | 0.85 | 0.82 |

XGBoost performs comparably to the best reproduced models and better than the paper's gradient boosting model.
With only 61 test patients, differences of 1–3 points correspond to one or two patients, so we do not claim it is
the single best model. See `notebooks/part2_xgboost.ipynb` for the full analysis.

## Reference

Shi, K., Bechler, K., Ho, V. *Predicting myocardial infarction risk using supervised and unsupervised machine
learning methods.*