import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


df = pd.read_csv("transformed_ebay_data.csv")

target_column = "num__Price" 
features = [col for col in df.columns if col != target_column]

X = df[features]
y = df[target_column]

# Identify Non-Numerical Columns
non_numeric_columns = X.select_dtypes(include=['object']).columns
print(f"Non-numeric columns: {non_numeric_columns}")

# Handle Non-Numeric Columns
# Drop irrelevant columns
columns_to_drop = [
    "remainder__Title",
    "remainder__Store URL",
    "remainder__Gallery URL",
    "remainder__Large Image URL",
    "remainder__Super Size Image URL",
    "remainder__View Item URL",
] 
X = X.drop(columns=columns_to_drop, errors="ignore")

# Encode categorical features
label_encoders = {}
for col in non_numeric_columns:
    if col not in columns_to_drop:  # Skip dropped columns
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Split the Data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train the XGBoost Model
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

eval_set = [(dtrain, "train"), (dval, "validation")]
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=eval_set,
    early_stopping_rounds=10,
    verbose_eval=True
)

# Evaluate the Model
y_pred = model.predict(dtest)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Feature Importance
xgb.plot_importance(model)
plt.show()
