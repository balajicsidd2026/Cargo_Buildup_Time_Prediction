# Generated from: code.ipynb
# Converted at: 2026-06-08T09:00:52.528Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


dataset=pd.read_csv("dataset/JED_Cargo_Export_Dataset.csv")
dataset

# **3.Basic Checkups**
# 


# Shape
print("Shape of the dataset:")
print(dataset.shape)
print("\n")

# Columns
print("Columns in the dataset:")
print(dataset.columns)
print("\n")

#check the null values in the dataset
print("Null values in the dataset:")
print(dataset.isnull().sum())



# Information about the dataset
print("Dataset Information:")
dataset.info()


#statistical summary of the dataset
print("Statistical Summary of the Dataset:")
dataset.describe()

# **4. Exploratory Data Analysis**


# 4.1 Target Distribution


target_column = "Build_Up_Time_Minutes"
plt.figure(figsize=(10,6))

sns.histplot(
    dataset[target_column],
    bins=30,
    kde=True
)

plt.title(
    f"Distribution of {target_column}"
)

plt.xlabel(target_column)

plt.ylabel("Frequency")

plt.show()

# 4.2 Weight vs Build-Up Time


plt.figure(figsize=(10,6))

sns.scatterplot(
    data=dataset,
    x='Cargo_Weight_KG',
    y='Build_Up_Time_Minutes'
)

plt.title('Weight vs Build-Up Time')
plt.xlabel('Weight (KG)')
plt.ylabel('Build-Up Time (Minutes)')

plt.show()

# 4.3 ULD Type vs Build-Up Time


plt.figure(figsize=(12,6))

sns.barplot(
    data=dataset,
    x='ULD_Type',
    y='Build_Up_Time_Minutes'
)

plt.title('ULD Type vs Build-Up Time')
plt.xticks(rotation=45)

plt.show() 

# 4.4 Cargo Category vs Build-Up Time


plt.figure(figsize=(12,6))

sns.barplot(
    data=dataset,
    x='Nature_of_Goods',
    y='Build_Up_Time_Minutes'
)

plt.title('Cargo Category vs Build-Up Time')
plt.xticks(rotation=45)

plt.show()

dataset.columns

# **5. Feature Engineering**


# 5.1 Data Splitting


# Features and Target

drop_columns = [
    'Build_Up_Time_Minutes',
    'Date',
    'Flight_ID'
]

X = dataset.drop(
    drop_columns,
    axis=1
)
y = dataset[
    'Build_Up_Time_Minutes'
]

# Train 70%
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# Validation 15%
# Test 15%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

# Backup for CatBoost

X_train_cat = X_train.copy()
X_val_cat = X_val.copy()
X_test_cat = X_test.copy()

y_train_cat = y_train.copy()
y_val_cat = y_val.copy()
y_test_cat = y_test.copy()

X_train_cat.info()

# 5.2 Encoding the categorical value


# Categorical Columns

categorical_cols = X_train.select_dtypes(
    include='object'
).columns

print(categorical_cols)

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(
        X_train[col]
    )
    X_val[col] = le.transform(
        X_val[col]
    )
    X_test[col] = le.transform(
        X_test[col]
    )
    label_encoders[col] = le

X_train

# 5.3 Scaling the numeric value


numerical_cols = X_train.columns

print(numerical_cols)

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(
    X_train_scaled[numerical_cols]
)

X_val_scaled[numerical_cols] = scaler.transform(
    X_val_scaled[numerical_cols]
)

X_test_scaled[numerical_cols] = scaler.transform(
    X_test_scaled[numerical_cols]
)

X_train_scaled.head()

# **6. Model Building**


# 6.1 Linear Regression


# Initialize Model
lr_model = LinearRegression()
lr_model.fit(
    X_train_scaled,
    y_train
)

y_pred_lr = lr_model.predict(
    X_val_scaled
)

print(
    "Linear Regression Model Trained Successfully"
)

# MAE
lr_mae = mean_absolute_error(y_val, y_pred_lr)

# MSE
lr_mse = mean_squared_error(y_val, y_pred_lr)

# RMSE
lr_rmse = np.sqrt(lr_mse)

# R2
lr_r2 = r2_score(y_val, y_pred_lr)

print("MAE :", lr_mae)
print("MSE :", lr_mse)
print("RMSE :", lr_rmse)
print("R2 Score :", lr_r2)

# 6.2 Decision Tree Regressor


# dt_param_grid = {
#     'max_depth': [5, 10, 15, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }

# dt_grid = GridSearchCV(
#     estimator=DecisionTreeRegressor(
#         random_state=42
#     ),
#     param_grid=dt_param_grid,
#     cv=5,
#     scoring='neg_mean_squared_error',
#     n_jobs=-1,
#     verbose=1
# )

# dt_grid.fit(
#     X_train,
#     y_train
# )

# print("Best Parameters:")
# print(dt_grid.best_params_)

# print("\nBest Score:")
# print(dt_grid.best_score_)

dt_model = DecisionTreeRegressor(
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

y_pred_dt = dt_model.predict(
    X_val
)

print(
    "Decision Tree Regressor Trained Successfully"
)

dt_mae = mean_absolute_error(
    y_val,
    y_pred_dt
)

dt_mse = mean_squared_error(
    y_val,
    y_pred_dt
)

dt_rmse = np.sqrt(
    dt_mse
)

dt_r2 = r2_score(
    y_val,
    y_pred_dt
)

print("MAE :", dt_mae)
print("MSE :", dt_mse)
print("RMSE :", dt_rmse)
print("R2 Score :", dt_r2)

# 6.3 Random Forest Regressor


# rf_param_grid = {
#     'n_estimators': [100, 200, 300, 500],
#     'max_depth': [5, 10, 15, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }

# rf_random = RandomizedSearchCV(
#     estimator=RandomForestRegressor(
#         random_state=42
#     ),
#     param_distributions=rf_param_grid,
#     n_iter=20,
#     cv=5,
#     scoring='neg_mean_squared_error',
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# rf_random.fit(
#     X_train,
#     y_train
# )

# print("Best Parameters:")
# print(
#     rf_random.best_params_
# )
# print("\nBest Score:")
# print(
#     rf_random.best_score_
# )

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features='log2',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

y_pred_rf = rf_model.predict(
    X_val
)

print(
    "Random Forest Regressor Trained Successfully"
)

rf_mae = mean_absolute_error(y_val,  y_pred_rf)
rf_mse = mean_squared_error(y_val, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_val, y_pred_rf)

print("MAE :", rf_mae)
print("MSE :", rf_mse)
print("RMSE :", rf_rmse)
print("R2 Score :", rf_r2)

# 6.4 XGBoost Regressor


# xgb_param_grid = {
#     'n_estimators': [100, 200, 300, 500],
#     'max_depth': [3, 5, 7, 10],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'min_child_weight': [1, 3, 5]
# }

# xgb_random = RandomizedSearchCV(
#     estimator=XGBRegressor(
#         random_state=42
#     ),
#     param_distributions=xgb_param_grid,
#     n_iter=20,
#     cv=5,
#     scoring='neg_mean_squared_error',
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# xgb_random.fit(
#     X_train,
#     y_train
# )

# print("Best Parameters:")
# print(
#     xgb_random.best_params_
# )

# print("\nBest Score:")
# print(
#     xgb_random.best_score_
# )

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=1,
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train
)

y_pred_xgb = xgb_model.predict(
    X_val
)

print(
    "XGBoost Regressor Trained Successfully"
)

xgb_mae = mean_absolute_error(
    y_val,
    y_pred_xgb
)

xgb_mse = mean_squared_error(
    y_val,
    y_pred_xgb
)

xgb_rmse = np.sqrt(
    xgb_mse
)

xgb_r2 = r2_score(
    y_val,
    y_pred_xgb
)

print("MAE :", xgb_mae)
print("MSE :", xgb_mse)
print("RMSE :", xgb_rmse)
print("R2 Score :", xgb_r2)

# 6.5 LightGBM Regressor


# lgbm_param_grid = {
#     'n_estimators': [100, 200, 300, 500],
#     'max_depth': [3, 5, 7, 10, 15],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'num_leaves': [15, 31, 63, 127],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0]
# }

# lgbm_random = RandomizedSearchCV(
#     estimator=LGBMRegressor(
#         random_state=42,
#         verbose=-1
#     ),
#     param_distributions=lgbm_param_grid,
#     n_iter=20,
#     scoring='neg_mean_squared_error',
#     cv=5,
#     random_state=42,
#     n_jobs=-1
# )

# lgbm_random.fit(X_train,y_train)

# print("Best Parameters:")
# print(lgbm_random.best_params_)

# print("\nBest Score:")
# print(lgbm_random.best_score_)

lgbm_model = LGBMRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.1,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42
)

lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_val)

print("LightGBM Regressor Trained Successfully")

lgbm_mae = mean_absolute_error(y_val,y_pred_lgbm)
lgbm_mse = mean_squared_error(y_val, y_pred_lgbm)
lgbm_rmse = np.sqrt(lgbm_mse)
lgbm_r2 = r2_score(y_val, y_pred_lgbm)

print("MAE :", lgbm_mae)
print("MSE :", lgbm_mse)
print("RMSE :", lgbm_rmse)
print("R2 Score :", lgbm_r2)

# 6.6 CatBoost Regressor


cat_features = X_train_cat.select_dtypes(
    include=['object','category']
).columns.tolist()

for col in cat_features:
    X_train_cat[col] = X_train_cat[col].astype('category')
    X_val_cat[col] = X_val_cat[col].astype('category')
    X_test_cat[col] = X_test_cat[col].astype('category')

# cat_param_grid = {
#     'iterations':[100,200,300,500],
#     'depth':[4,6,8,10],
#     'learning_rate':[0.01,0.05,0.1,0.2],
#     'l2_leaf_reg':[1,3,5,7,9]
# }

# cat_random = RandomizedSearchCV(
#     estimator=CatBoostRegressor(
#         random_state=42,
#         verbose=0
#     ),
#     param_distributions=cat_param_grid,
#     n_iter=20,
#     scoring='neg_mean_squared_error',
#     cv=5,
#     random_state=42,
#     n_jobs=-1
# )

# cat_random.fit(X_train_cat, y_train_cat,cat_features=cat_features)
# print("Best Parameters:")
# print(cat_random.best_params_)

# print("\nBest Score:")
# print(cat_random.best_score_)

cat_model = CatBoostRegressor(
    iterations=300,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=7,
    random_state=42,
    verbose=0
)

cat_model.fit(X_train_cat,y_train_cat,cat_features=cat_features)

y_pred_cat = cat_model.predict(X_val_cat)

print("CatBoost Regressor Trained Successfully")

cat_mae = mean_absolute_error(y_val_cat, y_pred_cat)
cat_mse = mean_squared_error(y_val_cat, y_pred_cat)
cat_rmse = np.sqrt(cat_mse)
cat_r2 = r2_score(y_val_cat, y_pred_cat)

print("MAE :", cat_mae)
print("MSE :", cat_mse)
print("RMSE :", cat_rmse)
print("R2 Score :", cat_r2)

# **7. Model Comparision**


model_results = pd.DataFrame({
    'Model': ['Linear Regression','Decision Tree', 'Random Forest','XGBoost', 'LightGBM','CatBoost'],
    'MAE': [lr_mae, dt_mae, rf_mae, xgb_mae, lgbm_mae, cat_mae],
    'MSE': [lr_mse, dt_mse, rf_mse, xgb_mse, lgbm_mse, cat_mse],
    'RMSE': [lr_rmse, dt_rmse, rf_rmse, xgb_rmse, lgbm_rmse, cat_rmse],
    'R2 Score': [lr_r2, dt_r2, rf_r2, xgb_r2, lgbm_r2, cat_r2]
})

model_results = model_results.sort_values(
    by='R2 Score',
    ascending=False
)

model_results

# **8. Save the Model**


import joblib

joblib.dump(
    xgb_model,
    'models/build_up_time_model.pkl'
)
joblib.dump(
    X_train.columns.tolist(),
    'models/feature_columns.pkl'
)
joblib.dump(
    label_encoders,
    'models/label_encoders.pkl'
)

print("Label Encoders Saved")
print("Feature Columns Saved")
print("Model Saved Successfully")

# **9. Test the model**


y_pred_test = cat_model.predict(X_test_cat)


test_mae = mean_absolute_error(y_test_cat,y_pred_test)
test_mse = mean_squared_error(y_test_cat, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test_cat,y_pred_test)


print("MAE :", test_mae)
print("MSE :", test_mse)
print("RMSE :", test_rmse)
print("R2 Score :", test_r2)


# **9. Feature Importance**


feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

feature_importance.head(20)