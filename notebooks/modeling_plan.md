# Modeling Plan

## Project Goal

The goal of this project is to predict long-term player retention for Steam games using information available at release or during approximately the first month after release.

The modeling target is:

`asinh_player_change = asinh(month12_avg_players) - asinh(month1_avg_players)`

This transformation was chosen instead of a raw retention ratio because many games have very small first-month player counts. A raw ratio can therefore become extremely unstable. The inverse hyperbolic sine (`asinh`) transformation remains well-defined near zero while behaving similarly to a logarithm for larger player counts.

Negative target values indicate declining player activity between month 1 and month 12, while positive values indicate growth.

The primary model-selection KPI is **Mean Absolute Error (MAE)**. RMSE and R-squared are reported as secondary metrics.

---

## Modeling Data

The final modeling table contains Steam metadata combined with monthly SteamCharts player-history data.

The main predictors are:

- Free-to-play status
- Initial USD price
- Number of supported languages
- Release year represented as years since 2015
- Windows support
- macOS support
- Linux support
- First-month average player activity transformed using `asinh`
- One-hot encoded Steam genres

The month-1 player feature is transformed because raw player counts are highly skewed, ranging from nearly zero to extremely large values.

Software-like applications were removed using Steam genre information before modeling.

Missing numerical values, especially launch price, are handled with median imputation inside the modeling pipeline. This ensures that imputation statistics are learned only from the corresponding training data rather than from future validation observations.

---

## Evaluation Strategy

A random train/test split was not used because the intended application is prediction for future Steam releases.

The target distribution also changes across release years, making temporal evaluation especially important.

All model families were therefore compared using the same expanding temporal validation folds:

| Fold | Training period | Validation year |
|---|---|---:|
| 1 | Through 2016 | 2018 |
| 2 | Through 2017 | 2019 |
| 3 | Through 2018 | 2020 |
| 4 | Through 2019 | 2021 |

A one-year gap is intentionally left between the end of the training period and the validation year. This reflects the fact that the month-12 outcome requires approximately one year after release to become observable.

Games released during 2022 and 2023 were reserved as an untouched out-of-time holdout.

The same information-availability rule was used for final evaluation:

- Predict 2022 games using training data through 2020.
- Predict 2023 games using training data through 2021.

The final holdout contained 109 games.

Games released in 2024 were not used in the primary final evaluation because the available SteamCharts history ends in September 2025, making the 2024 release cohort incomplete.

---

## Model Families Considered

The project intentionally used a limited set of defensible model families instead of testing every available algorithm.

### Dummy Median Regressor

The Dummy Median model predicts the median training target for every observation.

It establishes a minimum performance reference. Any useful predictive model should outperform this baseline consistently.

### Linear Regression

Linear Regression was used as the primary simple and interpretable model.

It tests whether the available launch and early-life features have an approximately linear relationship with long-term player change.

### Ridge Regression

Ridge Regression was tested to determine whether regularization could improve the temporal stability of the linear model.

The initial model used `alpha = 1` with standardized predictors.

Its results were nearly identical to ordinary Linear Regression, indicating that regularization at this level did not materially improve performance.

### Random Forest Regression

Random Forest was included because it can model nonlinear relationships and interactions without requiring explicit specification of those relationships.

This is particularly relevant for features such as first-month player count, price, genre, and release year, whose effects may not be purely linear.

### Gradient Boosting Regression

Gradient Boosting was included as a stronger tree-based ensemble capable of sequentially correcting model errors.

Because the untuned Gradient Boosting model produced the strongest average temporal cross-validation performance, it was selected for controlled hyperparameter tuning.

More complex alternatives such as neural networks, SVMs, XGBoost, LightGBM, and CatBoost were not pursued. The existing models were sufficient to compare linear and nonlinear approaches, and adding many additional algorithms would increase complexity without a clear modeling justification.

---

## Baseline Results

The initial model comparison across the four temporal validation folds produced approximately the following mean MAE values:

| Model | Mean MAE |
|---|---:|
| Gradient Boosting | 0.6430 |
| Random Forest | 0.6672 |
| Ridge Regression | 0.6771 |
| Linear Regression | 0.6772 |
| Dummy Median | 0.7159 |

Gradient Boosting produced the best average MAE.

Random Forest also showed strong temporal consistency and outperformed the Dummy Median baseline in every validation year.

Linear Regression performed very well in some cohorts but was less stable. In 2019 and 2020, its MAE was worse than the Dummy Median baseline.

Ridge Regression produced almost the same results as Linear Regression and did not materially improve temporal stability.

### Baseline MAE by Validation Year

| Validation year | Dummy Median | Linear Regression | Ridge | Random Forest | Gradient Boosting |
|---:|---:|---:|---:|---:|---:|
| 2018 | 0.7194 | 0.6160 | 0.6160 | 0.6971 | 0.6678 |
| 2019 | 0.7189 | 0.7261 | 0.7260 | 0.6616 | 0.6347 |
| 2020 | 0.6681 | 0.7140 | 0.7139 | 0.6365 | 0.6156 |
| 2021 | 0.7573 | 0.6527 | 0.6526 | 0.6735 | 0.6539 |

The variation across release years supports the use of temporal cross-validation rather than a single random split.

---

## Gradient Boosting Hyperparameter Tuning

Because Gradient Boosting had the lowest baseline mean MAE, it was selected for further tuning.

A deliberately small search grid was used rather than a large exhaustive search. This reduced the risk of overfitting hyperparameter decisions to a small number of temporal validation cohorts.

The following parameters were varied:

- `n_estimators`
- `learning_rate`
- `max_depth`

Every configuration was evaluated using exactly the same temporal folds.

The best configuration was:

```text
n_estimators = 100
learning_rate = 0.05
max_depth = 2
```

Its temporal cross-validation performance was:

| Metric | Value |
|---|---:|
| Mean MAE | 0.6369 |
| MAE standard deviation | 0.0363 |
| Mean RMSE | 0.8301 |
| Mean R-squared | 0.1022 |

### Best Configuration by Validation Year

| Validation year | MAE | RMSE | R-squared |
|---:|---:|---:|---:|
| 2018 | 0.6567 | 0.8655 | 0.1405 |
| 2019 | 0.6435 | 0.8612 | 0.1466 |
| 2020 | 0.5840 | 0.7594 | 0.0636 |
| 2021 | 0.6635 | 0.8343 | 0.0582 |

The tuned model improved modestly over the untuned Gradient Boosting baseline.

---

## Final Model Selection

The tuned Gradient Boosting model was selected as the final model based only on development-period temporal cross-validation.

This decision was made before evaluating the 2022-2023 holdout.

The selected configuration was:

```python
GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)
```

The model was chosen because it had the lowest mean MAE across the temporal validation folds while maintaining reasonably stable performance across years.

---

## Final Out-of-Time Evaluation

After model selection was complete, the selected Gradient Boosting model was evaluated on the untouched 2022-2023 holdout.

The final results were:

| Metric | Final Gradient Boosting |
|---|---:|
| Number of games | 109 |
| MAE | 0.7156 |
| RMSE | 0.9349 |
| R-squared | 0.0438 |

Performance was weaker than during development-period cross-validation.

The final holdout MAE was approximately 12% higher than the temporal CV mean MAE. This indicates that predicting player retention for later Steam release cohorts is harder than performance on earlier validation years would suggest.

The positive R-squared indicates that the model still retained some predictive information on future releases, although the explained variation was limited.

---

## Final Holdout Model Comparison

For diagnostic purposes, the other baseline models were also evaluated using the same final temporal holdout protocol.

| Model | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Random Forest | 0.6917 | 0.9274 | 0.0591 |
| Linear Regression | 0.6950 | 0.9186 | 0.0768 |
| Tuned Gradient Boosting | 0.7156 | 0.9349 | 0.0438 |
| Dummy Median | 0.8339 | 1.0724 | -0.2580 |

All three trained models substantially outperformed the Dummy Median model on the future holdout.

However, Random Forest achieved the lowest final MAE, while Linear Regression achieved the best final RMSE and R-squared.

This result shows that the additional complexity of tuned Gradient Boosting did not produce superior performance under the later temporal distribution shift.

The final model was **not changed after observing these results**. Gradient Boosting remains the model selected according to the predefined development-period evaluation strategy. Switching to Random Forest after examining the final holdout would use the test set for model selection and therefore compromise the independence of the final evaluation.

---

## What Did Not Work as Well

### Ridge Regression

Ridge Regression with `alpha = 1` produced almost identical results to ordinary Linear Regression.

This suggests that simple L2 regularization did not meaningfully address the main source of prediction error or temporal instability.

### Linear Model Stability

Linear Regression performed extremely well on some validation cohorts but worse than the Dummy Median model in others.

This suggests that the relationships between the current predictors and retention are not fully stable or linear across release periods.

### Additional Gradient Boosting Complexity

Increasing tree depth, increasing the number of estimators, or using a higher learning rate did not consistently improve validation MAE.

The best tuned model was relatively conservative, using depth-2 trees, 100 estimators, and a learning rate of 0.05.

### Final Temporal Generalization

Gradient Boosting performed best during temporal model selection but did not outperform Random Forest or Linear Regression on the final future cohort.

This suggests the project contains meaningful temporal distribution shift and that more model complexity does not automatically translate into better future performance.

---

## Interpretation

Feature importance from the selected Gradient Boosting model was saved under:

```text
results/interpretability/
```

These diagnostics indicate which variables the model relied on most strongly.

Feature importance should not be interpreted causally. A feature having high model importance means that it was useful for prediction within this dataset and modeling procedure, not that changing that feature would necessarily cause higher or lower player retention.

---

## Reproducibility

All modeling preprocessing and fitting are encapsulated in reusable scikit-learn pipelines.

The project includes:

```text
notebooks/
    modeling_baselines.ipynb
    modeling_experiments.ipynb

src/
    models/
        __init__.py
        train.py
        tune.py

results/
    baseline_cv_results.csv
    model_comparison.csv
    gradient_boosting_tuning_folds.csv
    gradient_boosting_tuning_summary.csv
    best_gradient_boosting_cv_results.csv
    final_holdout_predictions.csv
    final_holdout_results_by_year.csv
    final_model_comparison.csv
    interpretability/

artifacts/
    final_model.joblib
    final_model_metadata.json

tests/
    test_models.py
```

The serialized final model was successfully reloaded and used to produce predictions.

The automated model tests verify that:

- The modeling dataset loads.
- The required feature columns exist.
- The selected model fits and predicts successfully.
- Temporal cross-validation produces the expected fold structure.

All four model tests passed.

---

## Conclusion

The modeling experiments show that early-life Steam information contains measurable but limited signal for predicting long-term player activity.

Gradient Boosting produced the strongest performance during development-period temporal cross-validation, with a mean MAE of approximately 0.637, and was therefore selected as the final model.

On the untouched 2022-2023 holdout, its MAE increased to approximately 0.716 and R-squared declined to approximately 0.044. Random Forest and Linear Regression generalized slightly better on this future cohort.

These results suggest that temporal distribution shift is an important limitation of the current modeling problem. They also demonstrate that increased model complexity does not necessarily improve future predictive performance.

Future improvements should therefore focus primarily on richer early-life features, better characterization of changing Steam release cohorts, and further temporal robustness analysis rather than simply adding more complex model families.

---

## Acknowledgements

ChatGPT was used to assist with organizing the modeling workflow, structuring the evaluation plan, formatting and commenting Python code, troubleshooting Python and environment issues, and discussing model comparison, temporal validation, and reproducibility.
