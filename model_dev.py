"""model_dev.py

This module trains a supervised learning model to predict the insurance claim
probabilities based on historical customer data. It performs a feature selection
using a backward search aglorithim to remove noisy features and narrow it 
down to features that change the likely hood a customer would claim.

The goal is to build a model that does not over or under fit customer data
to create a accurate prediciton. 

Workflow:
    1. Load historical customer data and claim labels.
    2. Split the data into training and testing sets (80/20 split).
    3. Perform feature selection using backward search.
    4. Train an XGBoost classifier using the best features.
    5. Save the trained model and selected features to disk with joblib.

Dependencies:
    - pandas
    - numpy
    - xgboost
    - joblib
    - feature_selection (local module)
    - customer (local module)

Example:
    Run this file directly to train and save the final model:
        $ python model_dev.py

Output:
    "final_insurance_model.pkl"
"""

import pandas as pd
import numpy as np
from feature_selection import backward_search
from customer import Customer
import joblib
from xgboost import XGBClassifier

def customer_claim_probability(model, customer_data_subset):
    """Compute the probability of a claim for a subset of customer data.

    Uses a trained classification model to predict the probability that each
    customer in the given dataset will file an insurance claim.

    Args:
        model: A fitted machine learning model that supports `predict_proba()`.
        customer_data_subset (pd.DataFrame): A subset of customer features used
            for prediction.

    Returns:
        np.ndarray: A 1D array containing claim probabilities for each customer.
    """
    predictions = model.predict_proba(customer_data_subset)
    p_claim = predictions[:,1]
    return p_claim

if __name__ == "__main__":
    # load historical customer data
    customer_data = pd.read_csv('./data/customer_data.csv', index_col=None)
    customer_claims = pd.read_csv('./data/customer_claims.csv', index_col=None)

    # split: ~80% train / 20% test
    train_idx = np.random.random(size=customer_data.shape[0]) < 0.8

    x_train = customer_data.loc[train_idx]
    y_train = customer_claims['claim'].loc[train_idx]
    x_test  = customer_data.loc[~train_idx]
    y_test  = customer_claims['claim'].loc[~train_idx]
    print("text")
    best_features, best_acc = backward_search(x_train, y_train, x_test, y_test)  
    print("\n\nBest features found", best_features)
    print("Best accuracy found", best_acc, "\n")

    final_model = XGBClassifier(
        eval_metric='logloss',
        n_estimators=5000,          # allow many trees
        learning_rate=0.02,         # small step size
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(x_train[best_features], y_train)

    joblib.dump(final_model, 'final_insurance_model_truefinal.pkl')
    joblib.dump(best_features, 'best_model_features_truefinal.pkl')