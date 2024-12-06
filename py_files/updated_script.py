import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
df = pd.read_csv('transformed_data_fl.csv')

print(df)

df.isnull().sum()

df['Time duration'].describe()

plt.figure(figsize=(10, 6))  
plt.hist(df['Time duration'], bins=30, edgecolor='black')  

plt.title('Histogram of Sales__Time Duration')
plt.xlabel('Time Duration')
plt.ylabel('Frequency')

plt.show()

# calculate the quantiles
time_duration=df['Time duration']
quantiles = time_duration.quantile([0, 0.25, 0.5, 0.75, 1]).values.round().astype(int)
print("quantiles:", quantiles)

# Divide the time duration into 4 catergories.
df['category_timeDuration'] = pd.cut(df['Time duration'], 
                        bins=[0, 30, 90, 360, float('inf')], 
                        labels=['0-30 days', '30-90 days', '90-360 days', '>360 days'])

df = df.drop(['Shipping_Listing_Type_Encoded'],axis=1)

print(df)

# encode the label
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['y_label'] = label_encoder.fit_transform(df['category_timeDuration'])

df['y_label'].value_counts()

# retain the mappint relation of category_timeDuration & y_label
cateogory_timeduration_map = df.groupby('category_timeDuration')['y_label'].apply(lambda x:list(set(x))).to_dict()
print(cateogory_timeduration_map)

df = df.drop(['Time duration','category_timeDuration'],axis=1)

print(df)

df_nofeature_interaction =df.drop(['Cost_to_Price_Ratio','Feedback_Quality','Is_Trusted_Seller','Price_Condition'],axis=1)

df.describe()

print(df_nofeature_interaction)

# Initialize XGBoost and bulid Dmatrix for engeering_with Feature interaction

X = df.iloc[:,:-1]

y = df.iloc[:,-1]

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# trains an XGBoost model, retrieves the feature importances, and displays them in descending order.
# Train an initial XGBoost model
import xgboost as xgb
model_XGB_with_fi = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
model_XGB_with_fi.fit(X_train, y_train)

# Evaluate performance
score_with_fi = model_XGB_with_fi.score(X_test, y_test)
print("Model accuracy with feature interaction':", score_with_fi)

X_no_fi = df_nofeature_interaction.iloc[:,:-1]

y_no_fi = df_nofeature_interaction.iloc[:,-1]

# split X and y into training and testing sets

X_no_fi_train, X_no_fi_test, y_no_fi_train, y_no_fi_test = train_test_split(X_no_fi, y_no_fi, test_size = 0.2, random_state = 0)

model_XGB_no_fi = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
model_XGB_no_fi.fit(X_no_fi_train, y_no_fi_train)

# Evaluate performance
score_no_fi = model_XGB_no_fi.score(X_no_fi_test, y_no_fi_test)
print("Model accuracy without feature interaction:", score_no_fi)

# Get feature importance (with feature interaction)

feature_importances = model_XGB_with_fi.feature_importances_

# Combine features and their importance into a DataFrame for easy viewing
feature_importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': feature_importances
})

# Sort by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print(feature_importance_df)

# Drop the dominant feature
X_train_drop = X_train.drop(columns=['Listing Type_Encoded'])
X_test_drop = X_test.drop(columns=['Listing Type_Encoded'])

# bulid model with the data no list of type
model_XGB_with_fi_drop_listtype = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
model_XGB_with_fi_drop_listtype.fit(X_train_drop, y_train)

# Evaluate performance
score_no_dominant = model_XGB_with_fi_drop_listtype.score(X_test_drop, y_test)
print("Model accuracy without 'Listing Type_Encoded':", score_no_dominant)

feature_importances_drop = model_XGB_with_fi_drop_listtype.feature_importances_

# Combine features and their importance into a DataFrame for easy viewing
feature_importances_drop_df = pd.DataFrame({
    'feature': X_train_drop.columns,
    'importance': feature_importances_drop
})

# Sort by importance in descending order
feature_importances_drop_df = feature_importances_drop_df.sort_values(by='importance', ascending=False)
print(feature_importances_drop_df)

# why Listing Type_Encoded very important but no difference in accuracy after dropping it

# step1: statistical traits of the list_type

correlation = df[['Listing Type_Encoded', 'y_label']].corr()
print(correlation)

print(df['Listing Type_Encoded'].value_counts())
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='Listing Type_Encoded', y='y_label', data=df)
plt.title('Listing Type_Encoded vs y_label Distribution')
plt.show()

# step 2:Analyze the feature’s standalone effect

# Apply simple model by standalone feature
from sklearn.linear_model import LogisticRegression
X_single = X_train[['Listing Type_Encoded']]
model_single = LogisticRegression()
model_single.fit(X_single, y_train)
score_single = model_single.score(X_test[['Listing Type_Encoded']], y_test)
print("Accurary of standalone feature：", score_single)

# Evaluate feature interaction contributions
import shap
explainer = shap.TreeExplainer(model_XGB_with_fi)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)  # visulization of features and interaction

# the result could be: the categories of list_type is imbalanced, so 

# Compare the XGboost and RandomForest by default parameters

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


# Initialize the Random Forest model with default parameters
rf_model = RandomForestClassifier(random_state=42)

# Train the model on the training set
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# accuracy in training of randomforest after tuning
y_train_pred = rf_model.predict(X_train)  # Training predictions
# Evaluate the model's performance on training set
print("Training Set Performance:")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# Hyperparameter Tuning with Grid Search(random forest)

from sklearn.model_selection import GridSearchCV

#  Parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5]
                     }

#  Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best Parameters
print("Best Parameters:", grid_search.best_params_)

# Evalute by best parameters
best_rf = grid_search.best_estimator_
train_accuracy = accuracy_score(y_train, best_rf.predict(X_train))
test_accuracy = accuracy_score(y_test, best_rf.predict(X_test))

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# RandomForest_tuning1
# Define the Random Forest model with the best parameters
rf_model_tun1 = RandomForestClassifier(
    max_depth=15,
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=100,
    random_state=42
)

# Train the model on the training set
rf_model_tun1.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model_tun1.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# accuracy in training of randomforest after tuning
y_train_pred = rf_model_tun1.predict(X_train)  # Training predictions
# Evaluate the model's performance on training set
print("Training Set Performance:")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# RandomForest_tuning2
# Define the Random Forest model with the best parameters
rf_model_tun2 = RandomForestClassifier(
    max_depth=15,
    min_samples_leaf=20,
    min_samples_split=30,
    n_estimators=200,
    random_state=42
)

# Train the model on the training set
rf_model_tun2.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model_tun2.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# accuracy in training of randomforest after tuning
y_train_pred = rf_model_tun2.predict(X_train)  # Training predictions
# Evaluate the model's performance on training set
print("Training Set Performance:")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# RandomForest_tuning2
# Define the Random Forest model with the best parameters
rf_model_tun2 = RandomForestClassifier(
    max_depth=15,
    min_samples_leaf=20,
    min_samples_split=30,
    n_estimators=200,
    random_state=42
)

# Train the model on the training set
rf_model_tun2.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model_tun2.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# accuracy in training of randomforest after tuning
y_train_pred = rf_model_tun2.predict(X_train)  # Training predictions
# Evaluate the model's performance on training set
print("Training Set Performance:")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# RandomForest_tuning3
# Define the Random Forest model with the best parameters
rf_model_tun3 = RandomForestClassifier(
    max_depth=15,
    min_samples_leaf=30,
    min_samples_split=40,
    n_estimators=250,
    random_state=42
)

# Train the model on the training set
rf_model_tun3.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model_tun3.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# accuracy in training of randomforest after tuning
y_train_pred = rf_model_tun3.predict(X_train)  # Training predictions
# Evaluate the model's performance on training set
print("Training Set Performance:")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# RandomForest_tuning4
# Define the Random Forest model with the best parameters
rf_model_tun4 = RandomForestClassifier(
    max_depth=10,
    min_samples_leaf=40,
    min_samples_split=50,
    n_estimators=300,
    random_state=42
)

# Train the model on the training set
rf_model_tun4.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model_tun4.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# accuracy in training of randomforest after tuning
y_train_pred = rf_model_tun4.predict(X_train)  # Training predictions
# Evaluate the model's performance on training set
print("Training Set Performance:")
print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# Initialize the XGBoost model for multi-class classification


xgb_model = XGBClassifier(
    objective='multi:softmax',  # Use softmax for multi-class classification
    num_class=4,               # Specify the number of classes (4 in this case)
    use_label_encoder=False,   # Disable label encoder (for newer XGBoost versions)
    eval_metric='mlogloss',    # Multi-class log-loss evaluation metric
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score

# Define the number of folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize models
rf_model_test = RandomForestClassifier(random_state=42)
xgb_model_test = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Custom scorer for multi-class F1-score (weighted)
scorer = make_scorer(f1_score, average='weighted')

# Perform K-Fold Cross-Validation for Random Forest
rf_scores = cross_val_score(rf_model_test, X, y, cv=kf, scoring=scorer)
print(f"Random Forest - Average F1-Score: {rf_scores.mean():.4f} (Std: {rf_scores.std():.4f})")

# Perform K-Fold Cross-Validation for XGBoost
xgb_scores = cross_val_score(xgb_model_test, X, y, cv=kf, scoring=scorer)
print(f"XGBoost - Average F1-Score: {xgb_scores.mean():.4f} (Std: {xgb_scores.std():.4f})")

# pararmeters tuning
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier


# Configure the model:
# Declare parameters to reduce overfitting
# Cautions: the following parameters are just setted by intuition. we will tuin them next.
params = {
    'objective': 'multi:softmax',   # Specifies a multi-class classification objective
    'num_class': 4,                 # Specifies the number of classes in the target variable
    'max_depth': 8,                 # Limits the depth of each tree to control complexity
    'alpha': 10,                    # Adds L1 regularization to reduce overfitting
    'learning_rate': 0.3,           # Sets the learning rate to control the impact of each tree
    'n_estimators': 150             # Specifies the number of trees (boosting rounds)
}

# Instantiate the classifier with the specified parameters
xgb_tun1 = XGBClassifier(**params)

# Fit the classifier to the training data
xgb_tun1.fit(X_train, y_train)

# make predictions on test data
y_pred = xgb_tun1.predict(X_test)

# make predictions on test data
y_pred = xgb_tun1.predict(X_test)

# parameter tuning using GridSearchCV

# Convert the datasets into DMatrix format
dtrain = DMatrix(data=X_train, label=y_train)
dtest = DMatrix(data=X_test, label=y_test)

# Revised parameters for tuning
params_tun2 = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': 4,               # Number of classes
    'colsample_bytree': 0.3,      # Column sampling for each tree
    'learning_rate': 0.05,        # Lower learning rate for better generalization
    'max_depth': 15,              # Deeper trees
    'alpha': 5,                   # L1 regularization term
    'lambda': 1,                  # L2 regularization term
    'seed': 42                    # For reproducibility
}

# Define evaluation set (optional, for tracking progress)
evals = [(dtrain, 'train'), (dtest, 'test')]

# Train the model
best_num_boost_round = 1500  # Use the best number of boosting rounds obtained from previous tuning
XGB_model_tun2 = train(
    params=params_tun2,
    dtrain=dtrain,
    num_boost_round=best_num_boost_round,
    evals=evals,                    # Track training progress
    early_stopping_rounds=10        # Stop if no improvement after 10 rounds
)

# Make predictions on the test set
y_pred = XGB_model_tun2.predict(dtest)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Revised parameters for tuning
params_tun3 = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': 4,               # Number of classes
    'colsample_bytree': 0.7,      # Increase column sampling to improve generalization
    'subsample': 0.8,             # Sample rows for each boosting round
    'learning_rate': 0.03,        # Lower learning rate for better generalization
    'max_depth': 10,              # Balanced tree depth
    'min_child_weight': 5,        # Control complexity of the model
    'gamma': 1,                   # Minimum loss reduction to split a node
    'alpha': 5,                   # L1 regularization term
    'lambda': 2,                  # Increased L2 regularization
    'seed': 42                    # For reproducibility
}

# Define evaluation set for tracking progress
evals = [(dtrain, 'train'), (dtest, 'validation')]

# Train the model with early stopping
XGB_model_tun3 = train(
    params=params_tun3,
    dtrain=dtrain,
    num_boost_round=900,         # Start with a lower number of boosting rounds
    evals=evals,                  # Track training and validation progress
    early_stopping_rounds=20      # Stop if no improvement after 20 rounds
)

# Evaluate the model on the test set
preds_test = XGB_model_tun3.predict(dtest)
accuracy = accuracy_score(y_test, preds_test)
print(f"Test Set Accuracy: {accuracy:.4f}")

# Detailed classification metrics
print("Classification Report:")
print(classification_report(y_test, preds_test))

# Get feature importance from the trained model
feature_importance = XGB_model_tun3.get_score(importance_type='weight')
feature_importance_df = pd.DataFrame({
    'Feature': list(feature_importance.keys()),
    'Importance': list(feature_importance.values())
}).sort_values(by='Importance', ascending=False)

# Display feature importance
print("Feature Importance:")
print(feature_importance_df)

# Filter out less important features
threshold = 1000  # Set a threshold for importance
important_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature']

important_features = [col.strip() for col in important_features if col.strip() in X_train.columns]

print(important_features)

X_train_important = X_train.loc[:,important_features]
X_test_important = X_test.loc[:,important_features]

# Convert the datasets into DMatrix format
dtrain_important = DMatrix(data=X_train_important, label=y_train)
dtest_important = DMatrix(data=X_test_important, label=y_test)

# Define evaluation set (optional, for tracking progress)
evals_important = [(dtrain_important, 'train'), (dtest_important, 'test')]

model_drop_features = train(
    params=params_tun3,
    dtrain=dtrain_important,
    num_boost_round=900,
    evals=evals_important,
    early_stopping_rounds=20
)

# Evaluate the model on the test set
preds_test = model_drop_features.predict(dtest_important)
accuracy = accuracy_score(y_test, preds_test)
print(f"Test Set Accuracy: {accuracy:.4f}")

# Detailed classification metrics
print("Classification Report:")
print(classification_report(y_test, preds_test))

# Perform 5-fold cross-validation
cv_results = xgb.cv(
    params=params_tun3,
    dtrain=dtrain_important,
    num_boost_round=900,         # Maximum number of boosting iterations
    nfold=5,                      # Number of folds
    metrics={'mlogloss'},         # Metric for evaluation
    early_stopping_rounds=20,     # Stop if no improvement after 50 rounds
    stratified=True,              # Stratify folds by class distribution
    as_pandas=True,
    verbose_eval=10,              # Show log updates every 10 rounds
    seed=42                       # Reproducibility
)


# Display cross-validation results
print(cv_results)

best_iteration = len(cv_results)
best_mlogloss = cv_results['test-mlogloss-mean'].min()
print(f"Optimal number of boosting rounds: {best_iteration}")
print(f"Best mlogloss: {best_mlogloss:.4f}")

important_features = ['Feedback Score',
 'Category_Encoded',
 'Feedback_Quality',
 'Positive Feedback %',
 'Price_in_USD',
 'Price_Condition',
 'Is_Trusted_Seller',
 'Shipping Cost',
 'Cost_to_Price_Ratio',
 'Condition_Encoded',
 'Shipping Type_Encoded',
 'Store URL_flag',
 'Listing Type_Encoded']

X_train_important = X_train.loc[:,important_features]
X_test_important = X_test.loc[:,important_features]

# Convert the datasets into DMatrix format
dtrain_important = DMatrix(data=X_train_important, label=y_train)
dtest_important = DMatrix(data=X_test_important, label=y_test)

# Define evaluation set (optional, for tracking progress)
evals_important = [(dtrain_important, 'train'), (dtest_important, 'test')]

params_final = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': 4,               # Number of classes
    'colsample_bytree': 0.7,      # Increase column sampling to improve generalization
    'subsample': 0.8,             # Sample rows for each boosting round
    'learning_rate': 0.03,        # Lower learning rate for better generalization
    'max_depth': 10,              # Balanced tree depth
    'min_child_weight': 5,        # Control complexity of the model
    'gamma': 1,                   # Minimum loss reduction to split a node
    'alpha': 5,                   # L1 regularization term
    'lambda': 2,                  # Increased L2 regularization
    'seed': 42                    # For reproducibility
}

model = train(
    params=params_final,
    dtrain=dtrain_important,
    num_boost_round=900,
    evals=evals_important,
    early_stopping_rounds=20
)

import joblib

# Save model
joblib.dump(model, "model.pkl")

X_train_important


""" 
# Simulate a input symbol
This input is random but realistic, based on the ranges observed in your provided data. Here's how I generated each value:

Feedback Score: Chosen as 52000, within the observed range (between 417 and 3628114).
Category_Encoded: Random value of 1500, within a reasonable range for encoding.
Feedback_Quality: Randomly chosen as 1000, in line with typical values.
Positive Feedback %: A realistic value of 98.5%.
Price_in_USD: Set as $25, typical for mid-range pricing.
Price_Condition: Set to $20.000 to reflect a condition-adjusted price.
Is_Trusted_Seller: Set as 1 (True).
Shipping Cost: Randomly set as $5.00.
Cost_to_Price_Ratio: Chosen as 0.20, representing cost as a percentage of price.
Condition_Encoded: Set to 3, representing a condition level.
Shipping Type_Encoded: Random value of 1.
Store URL_flag: Set to 0, indicating no special store flag.
Listing Type_Encoded: Set as 1, representing a specific listing type.

"""

# Prepare the input as a dictionary
input_data = {
    "Feedback Score": [52000],
    "Category_Encoded": [1500],
    "Feedback_Quality": [1000],
    "Positive Feedback %": [98.5],
    "Price_in_USD": [25.00],
    "Price_Condition": [20.000],
    "Is_Trusted_Seller": [1],
    "Shipping Cost": [5.00],
    "Cost_to_Price_Ratio": [0.20],
    "Condition_Encoded": [3],
    "Shipping Type_Encoded": [1],
    "Store URL_flag": [0],
    "Listing Type_Encoded": [1],
}
# Convert the dictionary to a DataFrame
input_df = pd.DataFrame(input_data)

# forcast test
# load model
loaded_model = joblib.load("model.pkl")

# Convert DataFrame to DMatrix
dmatrix_input = xgb.DMatrix(input_df)
# forcast by the load_model
prediction = loaded_model.predict(dmatrix_input)
print(prediction)

# 1. SHAP Analysis
import shap
import matplotlib.pyplot as plt

# Create a SHAP explainer for the model
explainer = shap.TreeExplainer(loaded_model)

# 1. Global Feature Importance: Summary Plot (Dot Plot)
shap.summary_plot(shap_values.mean(axis=2), X_test_important)

# 2. Partial Dependence Plots (PDP)
from sklearn.inspection import PartialDependenceDisplay

# Define the indices or names of the features to plot
multiple_features = list(range(X_test_important.shape[1]))  # Select all features or specify indices (e.g., [0, 1, 2])

# Generate the partial dependence plots
PartialDependenceDisplay.from_estimator(
    loaded_model,  # Replace with your trained model
    X_test_important,
    multiple_features,
    feature_names=X_test_important.columns,
    grid_resolution=20
)

# Add a title and adjust layout
plt.suptitle("Partial Dependence Plots for Selected Features", fontsize=14)
plt.tight_layout()
plt.show()

import graphviz
print(graphviz.__version__)  # Check if it's successfully imported

from xgboost import plot_tree

# Set the figure size for better clarity
plt.figure(figsize=(60, 30))  # Adjust the size as needed

# Plot the first tree in the model
plot_tree(loaded_model, num_trees=0, rankdir='LR')  # Horizontal layout
plt.show()

# Save the tree as a high-resolution image
plt.figure(figsize=(20, 10))  # Ensure the saved figure has the same size
plot_tree(loaded_model, num_trees=0, rankdir='LR')
plt.savefig('xgboost_tree_structure_high_res.png', dpi=1000)  # Increase DPI for higher resolution

