# ============================================
# CROP RECOMMENDATION SYSTEM
# BYOP Project - Fundamentals of AI and ML
# ============================================

# STEP 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 50)
print("   CROP RECOMMENDATION SYSTEM")
print("=" * 50)

# ============================================
# STEP 2: Load the dataset
# Download from Kaggle: "Crop Recommendation Dataset"
# Save it as "crop_data.csv" in the same folder as this file
# ============================================

df = pd.read_csv("crop_data.csv")

print("\n✅ Dataset loaded successfully!")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ============================================
# STEP 3: Explore the data
# ============================================

print("\n--- First 5 rows of data ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Crops in dataset ---")
print(df['label'].unique())
print(f"Total crop types: {df['label'].nunique()}")

# ============================================
# STEP 4: Visualize the data
# ============================================

# Plot crop distribution
plt.figure(figsize=(14, 5))
df['label'].value_counts().plot(kind='bar', color='green', edgecolor='black')
plt.title('Number of samples per Crop')
plt.xlabel('Crop')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('crop_distribution.png')
plt.show()
print("\n✅ Chart saved as crop_distribution.png")

# Heatmap to see feature correlation
plt.figure(figsize=(10, 6))
sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap='YlGn', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("✅ Heatmap saved as correlation_heatmap.png")

# ============================================
# STEP 5: Prepare data for training
# ============================================

# Features (input) and Label (output)
X = df.drop('label', axis=1)   # Input: N, P, K, temperature, humidity, ph, rainfall
y = df['label']                 # Output: crop name

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Data split done!")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples : {X_test.shape[0]}")

# ============================================
# STEP 6: Train the model
# ============================================

print("\n⏳ Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ============================================
# STEP 7: Evaluate the model
# ============================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ============================================
# STEP 8: Make a prediction (try your own input!)
# ============================================

print("\n" + "=" * 50)
print("   TRY YOUR OWN PREDICTION")
print("=" * 50)

# You can change these values to test different conditions
sample_input = {
    'N': 90,            # Nitrogen
    'P': 42,            # Phosphorus
    'K': 43,            # Potassium
    'temperature': 20,  # in Celsius
    'humidity': 82,     # in %
    'ph': 6.5,          # soil pH
    'rainfall': 202     # in mm
}

input_df = pd.DataFrame([sample_input])
prediction = model.predict(input_df)

print(f"\nInput conditions: {sample_input}")
print(f"\n🌾 Recommended Crop: {prediction[0].upper()}")

# ============================================
# STEP 9: Feature importance
# ============================================

feature_names = X.columns
importances = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='teal')
plt.xlabel('Importance Score')
plt.title('Which factors matter most for crop selection?')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("\n✅ Feature importance chart saved!")

print("\n✅ Project complete! Check the generated charts.")
