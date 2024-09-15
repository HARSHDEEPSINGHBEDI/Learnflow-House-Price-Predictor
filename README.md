# LearnFlow - House Price Prediction

This repository contains a complete pipeline for predicting house prices using various machine learning models. The project includes data preprocessing, model training, evaluation, and generating predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Submission](#submission)

## Project Overview
This project involves using regression models to predict house prices. The goal is to build a model that can accurately estimate the sale price of a house based on various features.

## Data Description
The dataset used in this project includes:
- **Train Data**: Contains the features and target variable (`SalePrice`) for training the model.
- **Test Data**: Contains the features for which predictions need to be made.

The dataset may include missing values and irrelevant columns, which are addressed during preprocessing.

## Data Preprocessing

1. **Check for Missing Values**:
    ```python
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    ```

2. **Drop Irrelevant Columns**:
    ```python
    df_cleaned = df.drop(['Id', 'SomeIrrelevantColumn'], axis=1)
    ```

3. **Convert Categorical Columns**:
    ```python
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df_cleaned[['CategoricalColumn']])
    ```

4. **Prepare Data for Modeling**:
    ```python
    from sklearn.model_selection import train_test_split

    X = df_cleaned.drop('SalePrice', axis=1)
    y = df_cleaned['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

## Model Training

1. **Fit the Model**:
    ```python
    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    ```

2. **Generate Predictions**:
    ```python
    predictions = model.predict(X_test)
    ```

## Model Evaluation

1. **Evaluate Model Performance**:
    ```python
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    ```

Achieved an accuracy of around 65-70% on the training dataset with XGBoost. Due to time constraints, the model can be further improved.

## Submission

1. **Prepare Submission File**:
    ```python
    submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
    submission.to_csv('submission.csv', index=False)
    ```

Attached are the `Submission` and `Submission2` files.
