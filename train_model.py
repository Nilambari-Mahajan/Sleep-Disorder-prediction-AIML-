import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Split Blood Pressure into systolic and diastolic
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True).astype(int)

# Encode categorical columns
encoder = LabelEncoder()
categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

# Scale numerical features
scaler = StandardScaler()
numerical_columns = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                     'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Define features and target
X = data.drop(['Person ID', 'Blood Pressure', 'Sleep Disorder'], axis=1)
y = data['Sleep Disorder']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(rf_model, 'rf_sleep_model_with_bp.pkl')
