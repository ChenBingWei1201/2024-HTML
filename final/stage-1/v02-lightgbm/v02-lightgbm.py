import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb

# Load train and test data
train_data = pd.read_csv('./train_data.csv')
test_data = pd.read_csv('./same_season_test_data.csv')

# Identify non-predictive columns to drop
drop_cols = ['id', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher', 'date']

# Drop non-predictive columns from train data
X_train = train_data.drop(columns=[col for col in drop_cols if col in train_data.columns] + ['home_team_win'])
y_train = train_data['home_team_win']

# Drop non-predictive columns from test data
X_test = test_data.drop(columns=[col for col in drop_cols if col in test_data.columns])

# Feature Engineering: Add new features
X_train['rest_diff'] = X_train['home_team_rest'] - X_train['away_team_rest']
X_test['rest_diff'] = X_test['home_team_rest'] - X_test['away_team_rest']

# Example of interaction feature
X_train['batting_avg_diff'] = X_train['home_batting_batting_avg_10RA'] - X_train['away_batting_batting_avg_10RA']
X_test['batting_avg_diff'] = X_test['home_batting_batting_avg_10RA'] - X_test['away_batting_batting_avg_10RA']

# Encode categorical features
label_encoders = {}
for column in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])
    X_test[column] = le.transform(X_test[column])
    label_encoders[column] = le

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LightGBM model
lgbm = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=200, learning_rate=0.05)

# Hyperparameter tuning
param_grid = {
    'num_leaves': [31, 50],
    'max_depth': [-1, 10],
    'min_data_in_leaf': [20, 50],
    'feature_fraction': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate on training data
y_train_pred = best_model.predict(X_train_scaled)
accuracy = accuracy_score(y_train, y_train_pred)
roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train_scaled)[:, 1])

print(f"Training Accuracy: {accuracy:.4f}")
print(f"Training ROC AUC: {roc_auc:.4f}")

# Predict on test data
y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = y_test_pred_proba >= 0.5

# Prepare submission file
submission = pd.DataFrame({'id': test_data['id'], 'home_team_win': y_test_pred})
submission.to_csv('submission.csv', index=False)

print("Submission file created: 'submission.csv'")