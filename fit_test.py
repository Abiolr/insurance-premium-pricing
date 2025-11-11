from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier


model = joblib.load('final_insurance_model_truefinal.pkl')
features = joblib.load('best_model_features_truefinal.pkl')

customer_data = pd.read_csv('./data/customer_data.csv', index_col=None)
customer_claims = pd.read_csv('./data/customer_claims.csv', index_col=None)

train_idx = np.random.random(size=customer_data.shape[0]) < 0.8

x_train = customer_data.loc[train_idx]
y_train = customer_claims['claim'].loc[train_idx]
x_test  = customer_data.loc[~train_idx]
y_test  = customer_claims['claim'].loc[~train_idx]



y_all = customer_claims['claim'].astype(int)



#----------------------------- QUESTION 1 ----------------------------------------



y_proba = model.predict_proba(x_test)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

test_acc = accuracy_score(y_test, y_pred)
# create a 2x2 matrix of true negatives (TN), true positives (TP), false negatives (FN), and false positives(FP) 
# check page 39 in '6 Supervised Learning'
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()



#SENSITIVITY = TRUE POSITIVES/(TRUE POSITIVES + FALSE NEGATIVES)
#SPECIFICITY = TRUE NEGATIVES/(TRUE NEGATIVES + FALSE POSITIVES)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")




#--------------------------------QUESTION 2 ------------------------------------


# Compute accuracy on both training and testing data

train_predictions = model.predict(x_train[features])
test_predictions = model.predict(x_test[features])

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy:  {test_accuracy:.4f}")
