# Wildfire_Ignition_Forecasting
Wildfire ignition prediction in the United States using machine learning. Includes preprocessing, feature engineering, correlation/outlier analysis, and model training. A tuned XGBoost classifier achieves strong out-of-time performance with interpretable feature contributions.

`preprocess_feature_engineering.ipynb`
Processes the raw wildfire dataset (`WildFire_Data.csv`) together with external terrain data (not included due to size) and produces the final train and test files:

* `dataset_with_terrain_new_train_2014_2021.csv`
* `dataset_with_terrain_new_test_2022_2025.csv`

`corr_outlier_check.ipynb`
Performs correlation analysis and outlier inspection on the raw dataset, and both processed train and test datasets.

`baseline.ipynb`
Provides temperature baseline for model evaluation

`linear_regression.ipynb`, `random_forest.ipynb`, and `XGB_OF_baseline.ipynb`
Apply feature selection based on the earlier analysis, train the respective models, and output the evaluation results.

