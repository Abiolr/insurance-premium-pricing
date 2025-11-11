# Insurance Premium Pricing System

A machine learning-based system for predicting insurance claim probabilities, calculating customer premiums, and optimizing customer selection to maximize profitability while staying within coverage limits.

## Overview

This project implements an intelligent insurance underwriting system that:
- Predicts the probability of customers making insurance claims using XGBoost
- Calculates utility-based premiums tailored to each customer
- Selects optimal customers using a greedy knapsack algorithm
- Maximizes expected profit while managing risk exposure

## Project Structure

```
.
├── customer.py                 # Customer class with utility calculations
├── customer_selection.py       # Greedy algorithm for customer selection
├── feature_selection.py        # Backward feature selection algorithm
├── model_dev.py               # Model training and development
├── model_evaluation.py        # Premium setting and customer selection
├── fit_test.py                # Model performance evaluation
├── data/
│   ├── customer_data.csv      # Historical customer features
│   └── customer_claims.csv    # Historical claim outcomes
├── final_insurance_model_truefinal.pkl    # Trained XGBoost model
└── best_model_features_truefinal.pkl      # Selected features
```

## Key Components

### 1. Customer Class (`customer.py`)
Models individual insurance customers with:
- **Utility calculation**: Based on logarithmic utility function
- **Perceived utility**: Accounts for claim probability
- **Indifferent amount**: Maximum premium customer would accept
- **Premium calculation**: Weighted average of expected cost and indifferent amount
- **Profit calculation**: Premium minus expected cost

### 2. Machine Learning Model (`model_dev.py`)
- **Algorithm**: XGBoost Classifier
- **Feature Selection**: Backward search to identify predictive features
- **Training**: 80/20 train-test split with cross-validation
- **Output**: Claim probability predictions for risk assessment

### 3. Feature Selection (`feature_selection.py`)
Implements backward feature elimination:
- Iteratively removes features that don't improve accuracy
- Uses accuracy score as the evaluation metric
- Prevents overfitting by identifying truly predictive features

### 4. Customer Selection (`customer_selection.py`)
Greedy knapsack algorithm that:
- Prioritizes customers with lower coverage amounts
- Maximizes number of customers within $100M coverage limit
- Ensures portfolio diversification

### 5. Premium Setting (`model_evaluation.py`)
End-to-end pipeline that:
- Loads trained model and predicts claim probabilities
- Creates Customer objects with calculated premiums
- Filters customers based on competitiveness (top 5% premium excluded)
- Filters customers based on profitability (bottom 35% profit excluded)
- Selects optimal customer portfolio using greedy algorithm

## Installation

### Requirements
```bash
pip install numpy pandas scikit-learn xgboost joblib
```

### Dependencies
- Python 3.8+
- numpy
- pandas
- scikit-learn
- xgboost
- joblib

## Usage

### Training the Model
```python
python model_dev.py
```
This will:
1. Load historical customer and claim data
2. Perform feature selection
3. Train the XGBoost model
4. Save the model and features to disk

### Evaluating Model Performance
```python
python fit_test.py
```
Outputs:
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- Training and testing accuracy

### Setting Premiums and Selecting Customers
```python
python model_evaluation.py
```
This generates a DataFrame with:
- `p_claim`: Predicted claim probability for each customer
- `desired_customer`: Boolean indicating selected customers
- `premium`: Calculated premium amount

## Model Performance Metrics

The system evaluates model performance using:
- **Accuracy**: Overall prediction correctness
- **Sensitivity**: Ability to identify customers who will claim
- **Specificity**: Ability to identify customers who won't claim
- **Confusion Matrix**: Breakdown of true/false positives/negatives

## Algorithm Details

### Utility Function
```
U(x) = -263.31 + 22.09 × ln(x + 150,000)
```
Where `x` is the dollar amount (positive for gains, negative for losses).

### Premium Calculation
```
Premium = 0.75 × Expected_Cost + 0.25 × Indifferent_Amount
```
This balances the insurer's expected payout with the customer's willingness to pay.

### Expected Cost
```
Expected_Cost = P(claim) × Coverage_Amount
```

## Selection Criteria

Customers are filtered based on:
1. **Premium competitiveness**: Exclude top 5% highest premiums
2. **Profitability threshold**: Exclude bottom 35% profits
3. **Coverage constraint**: Total coverage ≤ $100,000,000

## Key Features

- **Utility-Based Pricing**: Premiums reflect both risk and customer value perception
- **Machine Learning**: XGBoost provides accurate claim probability predictions
- **Feature Selection**: Removes noise and improves model generalization
- **Portfolio Optimization**: Maximizes customers while respecting coverage limits
- **Risk Management**: Filters out unprofitable and overpriced policies

## License

This project is part of an academic assignment and is intended for educational purposes.