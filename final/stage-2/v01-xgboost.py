import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load train and test data
train_data = pd.read_csv('./train_data.csv')
test_data = pd.read_csv('./2024_test_data.csv')

# Identify columns to drop (non-predictive or identifiers)
drop_cols = [
    'id', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 
    'away_pitcher', 'date', 'season', 'home_team_season', 'away_team_season'
]

# Prepare train features and target
X_train = train_data.drop(columns=[col for col in drop_cols if col in train_data.columns] + ['home_team_win'])
y_train = train_data['home_team_win']

# Prepare test features
X_test = test_data.drop(columns=[col for col in drop_cols if col in test_data.columns])

# Feature Engineering: Add new features
X_train['rest_diff'] = X_train['home_team_rest'] - X_train['away_team_rest']
X_test['rest_diff'] = X_test['home_team_rest'] - X_test['away_team_rest']

X_train['pitcher_rest_diff'] = X_train['home_pitcher_rest'] - X_train['away_pitcher_rest']
X_test['pitcher_rest_diff'] = X_test['home_pitcher_rest'] - X_test['away_pitcher_rest']

# Encode categorical features
label_encoders = {}
for column in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])
    X_test[column] = le.transform(X_test[column])
    label_encoders[column] = le

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
}

xgb = XGBClassifier(eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate on training data
y_train_pred = best_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train_scaled)[:, 1])

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training ROC AUC: {train_roc_auc:.4f}")

# Predict on test data
y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = y_test_pred_proba >= 0.5

# Prepare submission file
submission = pd.DataFrame({'id': test_data['id'], 'home_team_win': y_test_pred})
submission.to_csv('./submission.csv', index=False)

print("Submission file created: 'submission.csv'")