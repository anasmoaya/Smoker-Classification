import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Load the datasets
train_df = pd.read_csv('/kaggle/input/emsi-tabular-2024/train.csv')
test_df = pd.read_csv('/kaggle/input/emsi-tabular-2024/test.csv')

# Adding new features -
for df in [train_df, test_df]:
    df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
    df['HW_Ratio'] = df['height(cm)'] / df['waist(cm)']
    df['HA_Ratio'] = df['height(cm)'] / df['age']
    # Consider adding or modifying features

# Removing less relevant features 
features_to_remove = ['eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)', 'dental caries' , 'Urine protein']
train_df.drop(columns=features_to_remove, inplace=True)
test_df.drop(columns=features_to_remove, inplace=True)

# Update feature list - MODIFY HERE
features_to_include = list(set([feature for feature in train_df.columns if feature not in ['id', 'smoking']]))

# Separating features and target variable in the training dataset
X = train_df[features_to_include]
y = train_df['smoking']

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a preprocessing pipeline for numerical features - MODIFY HERE
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Try 'median' or 'most_frequent'
    ('scaler', StandardScaler()) # Try MinMaxScaler() or RobustScaler()
])

# Preprocessor for the updated features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, features_to_include)
    ])

# Pipeline with preprocessing and Gradient Boosting classifier - MODIFY HERE
gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', GradientBoostingClassifier(
                                  random_state=42,
                                  n_estimators=1000, # Try modifying
                                  learning_rate=0.2, # Try modifying
                                  # Add other hyperparameters here
                                  ))])

# Fit the model on the training data
gb_pipeline.fit(X_train, y_train)

# Predict probabilities on the validation data
pred_prob = gb_pipeline.predict_proba(X_val)[:, 1]

# Compute AUC-ROC on validation set
auc_roc_score = roc_auc_score(y_val, pred_prob)
print("AUC-ROC Score on Validation Set:", auc_roc_score)

# Apply the model to the test data
X_test = test_df[features_to_include]
test_predictions = gb_pipeline.predict_proba(X_test)[:, 1]

# Creating a DataFrame for submission
submission_df = pd.DataFrame({
    'id': test_df['id'], 
    'smoking_probability': test_predictions
})

# Save the submission file to the specified directory
submission_path = '/kaggle/working/submission.csv'
submission_df.to_csv(submission_path, index=False)

print("Submission file saved to:", submission_path)
print(submission_df.head())
