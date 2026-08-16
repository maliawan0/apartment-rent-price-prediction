# Apartment Rent — Classification & Price Prediction

Final Data Mining project on the **Apartments for Rent Classified** dataset (UCI). Two modelling tasks on the same data: classify listings into categories, and regress the actual rent price.

## The pipeline

1. **Preprocessing** — missing-value handling, categorical encoding, feature scaling
2. **Exploratory analysis** — distributions, correlations, geographic spread
3. **Classification** — five algorithms compared on the same train/test split
4. **Clustering** — unsupervised structure in the listings
5. **Regression** — rent price prediction
6. **Evaluation** — accuracy, precision/recall, confusion matrices; error metrics for the regressor

## Models

**Classification**

| Model | Note |
|---|---|
| Decision Tree | Interpretable baseline |
| Random Forest | Ensemble — best classification performance |
| Support Vector Machine | Tested across kernels |
| k-Nearest Neighbours | Sensitive to feature scaling |
| Naive Bayes | Fast probabilistic baseline |

**Clustering**

| Model | Note |
|---|---|
| K-Means | Centroid-based, k selected by elbow method |
| DBSCAN | Density-based — finds irregular clusters and flags outliers |

**Regression** — rent price predicted from listing features.

## Layout

```
Apartment-For-Rent-Project.ipynb   # the full analysis, end to end
```

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Apartment-For-Rent-Project.ipynb
```

## Stack

Python · pandas · scikit-learn · matplotlib / seaborn · Jupyter
