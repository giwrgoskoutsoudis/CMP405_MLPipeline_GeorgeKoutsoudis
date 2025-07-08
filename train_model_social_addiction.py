# -----------------------------------------
# 📦 Βήμα 1: Εισαγωγή Βιβλιοθηκών
# -----------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE  # εγκατέστησε με: pip install imbalanced-learn
import joblib

# -----------------------------------------
# 📂 Βήμα 2: Φόρτωση Dataset
# -----------------------------------------
url = "https://raw.githubusercontent.com/giwrgoskoutsoudis/AI-Project/main/Students%20Social%20Media%20Addiction.csv"
df = pd.read_csv(url)

# -----------------------------------------
# 🧼 Βήμα 3: Καθαρισμός & Μετατροπή Target
# -----------------------------------------
def categorize(score):
    return "Low" if score <= 5 else "High"

df["Addicted_Level"] = df["Addicted_Score"].apply(categorize)

if "Student_ID" in df.columns:
    df.drop("Student_ID", axis=1, inplace=True)

# -----------------------------------------
# 🔠 Βήμα 4: Κωδικοποίηση Κατηγορικών
# -----------------------------------------
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# -----------------------------------------
# 🎯 Βήμα 5: Επιλογή Χαρακτηριστικών
# -----------------------------------------
selected_features = [
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Affects_Academic_Performance",
    "Conflicts_Over_Social_Media"
]

X = df[selected_features]
y = df["Addicted_Level"]

# -----------------------------------------
# 🧪 Βήμα 6: Train/Test Split + SMOTE
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# -----------------------------------------
# 🤖 Βήμα 7: Εκπαίδευση Μοντέλου
# -----------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# -----------------------------------------
# 📊 Βήμα 8: Αξιολόγηση Μοντέλου
# -----------------------------------------
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("🔵 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------------------
# 💾 Βήμα 9: Αποθήκευση Μοντέλου & LabelEncoder
# -----------------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("📁 Τα αρχεία αποθηκεύτηκαν: model.pkl & label_encoder.pkl")
