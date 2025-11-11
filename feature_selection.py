"""feature_selection.py

This module implements a backward feature selection algorithm that iteratively
removes features to improve model performance. 

Typical workflow:
    1. Train a model on all features.
    2. Iteratively remove one feature at a time.
    3. Keep the change if removing a feature improves accuracy.
    4. Stop when no further improvement is achieved.

Dependencies:
    - xgboost
"""

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def backward_search(x_train, y_train, x_test, y_test):
    """Perform backward feature selection to maximize model accuracy.

    The algorithim starts with a list of all features and removes the 
    features that show no increase on model accuracy.
    The search terminates when no more increase in accuracy is seen.

    Args:
        x_train (pd.DataFrame): Training dataset containing feature columns.
        y_train (pd.Series): Training labels corresponding to `x_train`.
        x_test (pd.DataFrame): Test dataset containing the same features as `x_train`.
        y_test (pd.Series): Test labels corresponding to `x_test`.

    Returns:
        tuple:  
            - best_features (list[str]): List of selected features that yield the
              best accuracy.
            - best_acc (float): The highest test accuracy achieved during the search.

    Notes:
        - By default, the model used is an `XGBClassifier` with log-loss evaluation.
        - You can swap in other models (e.g., `LogisticRegression` or
          `RandomForestClassifier`) by uncommenting the relevant lines.
        - The algorithm uses a small threshold (0.0001) to prevent overreacting
          to minor fluctuations in accuracy due to randomness.
    """
    model = XGBClassifier(
        eval_metric='logloss',   # tells XGBoost to use log-loss for classification
        n_estimators=10000,      # use this, not max_iter
        learning_rate=0.01,      # slower learning, more stable over many trees
        max_depth=5,             # controls model complexity
        subsample=0.8,           # random subsampling of rows (reduces overfitting)
        colsample_bytree=0.8,    # random subsampling of features per tree
        random_state=42,
        n_jobs=-1,               # uses all CPU cores
    )

    all_columns = list(x_train.columns) 
    best_features = list(all_columns)
    x_train_subset = x_train[best_features]
    x_test_subset = x_test[best_features]
    model.fit(x_train_subset, y_train)
    pred = model.predict(x_test_subset)
    best_acc = accuracy_score(y_test, pred)
    threshold = 0.0001
    
    while True:
        feature_to_remove = None
        best_acc_this_round = best_acc

        for f in best_features:
            current_features_list = [feat for feat in best_features if feat != f]

            if not current_features_list:
                continue

            x_train_subset = x_train[current_features_list]
            x_test_subset  = x_test[current_features_list]

            model.fit(x_train_subset, y_train)
            pred = model.predict(x_test_subset)
            curr_acc = accuracy_score(y_test, pred)

            if (curr_acc > best_acc_this_round + threshold):
                best_acc_this_round = curr_acc
                feature_to_remove = f

        # after testing all features remove the one that gave best improvement
        if feature_to_remove is not None:
            best_features.remove(feature_to_remove)
            best_acc = best_acc_this_round
        else:
            # no improvement stop
            break

    return best_features, best_acc