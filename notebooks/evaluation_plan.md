# Evaluation Plan

## Project Overview

The goal of this project is to determine whether information available during the early life of a newly released Steam game can be used to predict its long-term player retention.

The main research question is:

**Can information available during approximately the first month after a Steam game's release predict how well the game retains its player base approximately one year later?**

The intended users of this analysis are game developers and publishers who may want to understand which characteristics of a game's launch and early reception are associated with sustained player engagement.

---

## Unit of Analysis

The unit of analysis is one Steam game.

Each observation in the modeling dataset represents a single game and includes:

- Basic Steam metadata
- Price information
- Platform support
- Supported languages
- Genre information
- Early player activity
- Player activity approximately one year later

---

## Data Sources

The project currently combines two main data sources.

### Steam Dataset 2025

This dataset provides Steam application metadata, including:

- App ID
- Game name
- Release date
- Free-to-play status
- Initial price
- Operating-system support
- Supported languages
- Genres
- Additional Steam metadata

Only applications labeled as games are retained.

The dataset is further restricted to games released from 2015 through 2024.

Software-like applications identified through genres such as Utilities, Game Development, Video Production, and Design & Illustration are removed from the current modeling population.

### Historical Player Data

Historical monthly average and peak player counts are obtained from a public dataset containing Steam player histories.

Games are matched between the two datasets using Steam `appid`.

The current combined modeling dataset contains **276 games** for which both Steam metadata and the required historical player-count observations are available.

---

## Current Dataset Limitations

The current historical player dataset is not a representative sample of all Steam releases.

It consists primarily of multiplayer and relatively popular games.

Therefore, the current results should be treated as an initial modeling experiment rather than a general conclusion about all Steam games.

Expanding the historical player-count dataset is one of the most important future improvements.

---

## Feature Engineering

Several features are constructed from the original Steam data.

### Price

Steam price values are stored in cents.

For games whose currency is USD, the original price is converted to dollars:

`initial_price_usd = mat_initial_price / 100`

Non-USD prices are currently treated as missing rather than converted using exchange rates.

Missing prices are handled later using median imputation inside the machine-learning pipeline.

Initial price is preferred to current or discounted price because the project aims to use information available near the beginning of a game's life.

### Supported Languages

The original supported-language field is stored as text and may contain HTML formatting.

The text is cleaned and converted into a numerical feature:

`language_count`

This represents the number of languages supported by the game.

### Genres

The Steam genre tables are joined to the main application table using `appid`.

Common genres are converted into binary indicator variables such as:

- `genre_Action`
- `genre_Adventure`
- `genre_Casual`
- `genre_Early_Access`
- `genre_Free_To_Play`
- `genre_Indie`
- `genre_Massively_Multiplayer`
- `genre_RPG`
- `genre_Racing`
- `genre_Simulation`
- `genre_Sports`
- `genre_Strategy`

A value of 1 indicates that a game belongs to that genre, while 0 indicates that it does not.

Rare genres are excluded from the current baseline feature set.

---

## Player Retention Definition

The project measures player activity using average concurrent-player counts.

The first-month value is defined as the first full calendar month after the game's release.

The long-term value is taken approximately 12 months after release.

The raw retention ratio is:

`retention_ratio = month12_avg_players / month1_avg_players`

For example:

- A retention ratio of `0.50` means that average player activity after approximately one year is 50% of the first-month average.
- A retention ratio of `1.00` means activity is approximately unchanged.
- A retention ratio greater than `1.00` means that the game has more average players after approximately one year than during its first full month.

---

## Modeling Target

The raw retention ratio can contain extreme values, especially when a game begins with a very small player population.

For this reason, the current regression target is:

`log_retention = log((month12_avg_players + 1) / (month1_avg_players + 1))`

The logarithmic transformation reduces the influence of very large retention ratios and makes the regression problem more stable.

---

## Current Predictors

The current baseline models use the following features:

- Free-to-play status
- Initial USD price
- Number of supported languages
- Windows support
- Mac support
- Linux support
- Average players during the first full month
- Common Steam genre indicators

Additional features may be added later, including:

- Steam categories
- Multiplayer and co-op functionality
- Developer history
- Publisher history
- Early Steam review information
- Early YouTube attention
- Early review sentiment
- Other launch-related features

---

## Prediction Setting

The intended prediction takes place approximately one month after a game's release.

Therefore, predictors should only contain information that would have been available by that time.

The project should not use information from later in the game's life to predict long-term retention.

---

## Temporal Train, Validation, and Test Split

A random train-test split is not used.

The project instead uses a temporal split so that older games are used to predict newer releases.

The current split is:

- **Training set:** games released from 2015 through 2020
- **Validation set:** games released in 2021 and 2022
- **Final test set:** games released in 2023 and 2024

The resulting sizes are:

- Training: 209 games
- Validation: 49 games
- Test: 18 games

The validation set is used for comparing models and making modeling decisions.

The 2023–2024 test set is reserved as a final holdout and has not yet been used for model selection.

Because the final test set contains only 18 games, its eventual performance estimate will have relatively high uncertainty.

---

## Why a Temporal Split Is Important

A temporal split more closely reflects the intended real-world use of the model.

In practice, a developer or publisher would want to use information about previously released games to make predictions about a new release.

Using newer games to predict older games would create an unrealistic evaluation and could introduce temporal leakage.

A future version of the project may use rolling temporal cross-validation rather than a single temporal holdout.

---

## Leakage Risks

Leakage is an important concern because many Steam variables continue to change after release.

The following variables must not be used as predictors if they occur after the first-month prediction point:

- Month-12 player counts
- Retention ratio
- Log-retention target
- Player counts measured after the first month
- Reviews written later in the game's life
- Current review totals
- Later YouTube coverage
- Later discounts
- Current sale prices
- DLC released later
- Major updates released later
- Developer performance from games released after the prediction date
- Publisher performance from later releases

The target variables are used only for evaluation and are never included as predictors.

---

## Missing Data

Some predictor variables contain missing values.

For example, some games do not have usable USD price information.

Missing numerical values are handled with median imputation.

The imputer is placed inside a scikit-learn `Pipeline`.

This is important because the median is learned only from the training data rather than from the full dataset, reducing the risk of data leakage.

---

## Baseline Models

Three baseline regression models are currently evaluated.

### Dummy Regressor

The Dummy Regressor predicts the median target value from the training data for every game.

It does not use any of the game features.

This provides a simple baseline that real models should attempt to outperform.

### Linear Regression

Linear Regression uses the available predictor variables and learns a linear relationship between the features and log retention.

This provides a simple interpretable baseline.

### Random Forest Regressor

Random Forest is used as a nonlinear baseline.

The current model uses 200 trees and a fixed random seed.

No major hyperparameter tuning has been performed yet because the current goal is to establish a baseline evaluation pipeline rather than optimize final performance.

---

## Evaluation Metrics

The primary evaluation metric is **Mean Absolute Error (MAE)**.

MAE measures the average absolute difference between the predicted and actual log-retention values.

It is selected as the primary metric because it is less sensitive to extreme prediction errors than squared-error metrics.

Two secondary metrics are also reported.

### Root Mean Squared Error

RMSE gives more weight to large prediction errors.

### R²

R² measures how much of the variation in the target is explained by the model relative to a constant mean prediction.

Negative R² values are possible and indicate that the model performs worse than predicting the validation-set mean.

---

## Baseline Validation Results

The current validation results are:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Dummy Median | 0.7773 | 0.9604 | -0.2654 |
| Linear Regression | 0.7630 | 0.9252 | -0.1742 |
| Random Forest | 0.7376 | 0.8943 | -0.0971 |

Random Forest currently has the best validation performance of the three models.

Both Linear Regression and Random Forest improve upon the Dummy Regressor.

However, the negative R² values indicate that the models currently have limited predictive power.

These results should therefore be interpreted as baseline results rather than final model performance.

---

## Shuffled-Target Stress Test

A shuffled-target stress test is used to determine whether the Random Forest is learning a genuine relationship between the predictors and retention.

The training target is randomly shuffled while the feature matrix remains unchanged.

This destroys the real relationship between individual games and their retention values.

The Random Forest is then retrained using the shuffled target and evaluated against the original validation target.

The results are:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Random Forest | 0.7376 | 0.8943 | -0.0971 |
| Random Forest with shuffled target | 0.8656 | 1.1313 | -0.7557 |

Performance becomes substantially worse after shuffling the target.

This suggests that the original model is learning some genuine structure in the data rather than relying entirely on artifacts in the modeling pipeline.

However, overall predictive performance is still weak and requires improvement.

---

## Overfitting and Model Selection

The validation set is used for model comparison.

The final test set should not be used for:

- Feature selection
- Model selection
- Hyperparameter selection
- Repeated experimentation

Once a final modeling approach has been selected, it can be evaluated once on the untouched test set.

If more advanced hyperparameter tuning is performed later, it should use training data only or an appropriate temporal cross-validation procedure.

---

## Additional Stress Tests Planned

Several additional robustness tests may be performed later.

### Remove First-Month Player Count

The model can be retrained without `month1_avg_players`.

This will help determine how much of the predictive performance comes simply from early popularity versus other game characteristics.

### Temporal Stability

Model performance can be evaluated across different release-year cohorts.

This can reveal whether relationships learned from older Steam games continue to hold for newer games.

### Unseen Developers

If developer information is added later, evaluation can specifically examine games from developers that did not appear in the training data.

### Outlier Sensitivity

Retention and player-count distributions may contain extreme values.

Future evaluation may compare performance with and without extreme observations or use additional transformations and robust metrics.

---

## Current Limitations

The project currently has several important limitations.

### Small Sample Size

The modeling dataset contains only 276 games.

This limits model complexity and makes validation estimates less stable.

### Selection Bias

The historical player dataset focuses primarily on multiplayer and relatively popular games.

The current results may therefore not generalize to the broader Steam catalog.

### Small Final Test Set

The final test set contains only 18 games.

The eventual final-test metrics will have substantial uncertainty.

### Limited Feature Set

The baseline models currently use only a subset of the features that may influence retention.

Potentially useful variables such as reviews, developer reputation, publisher history, multiplayer categories, updates, and external attention have not yet been incorporated.

### Historical Data Availability

Historical launch-time information is difficult to reconstruct.

Any future feature added to the project must be checked carefully to ensure that it would actually have been available at the prediction time.

---

## Planned Improvements

Future work may include:

- Expanding the historical player-count dataset
- Increasing the number of games in the modeling sample
- Adding Steam categories
- Adding multiplayer and co-op indicators
- Adding developer history
- Adding publisher history
- Adding early Steam review information
- Adding early review sentiment
- Adding early YouTube attention
- Improving price handling
- Testing additional regression models
- Performing hyperparameter tuning
- Implementing rolling temporal cross-validation
- Performing more stress tests
- Evaluating feature importance
- Comparing models with and without first-month player count
- Improving the final test-set evaluation once more historical data are available

---

## Current Conclusion

The current baseline evaluation shows that the available features contain some predictive information about long-term player retention.

Linear Regression performs slightly better than the Dummy Regressor, and Random Forest performs best among the three current baseline models.

The shuffled-target stress test produces substantially worse performance, providing additional evidence that the original model is learning some real signal.

However, all current models still have negative validation R² values, so the project should not yet be considered strongly predictive.

The next stage of the project will focus on improving data coverage, adding relevant launch-time features, and strengthening the temporal evaluation procedure.

---

## Acknowledgements

ChatGPT was used as a supporting tool during the development of this project.
