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

from imblearn.over_sampling import SMOTE

# -----------------------------------------
# 📂 Βήμα 2: Φόρτωση Dataset
# -----------------------------------------
url = "https://raw.githubusercontent.com/giwrgoskoutsoudis/AI-Project/main/Students%20Social%20Media%20Addiction.csv"
df = pd.read_csv(url)

# -----------------------------------------
# 👀 Βήμα 3: Εξερεύνηση
# -----------------------------------------
print("📐 Shape:", df.shape)
print("🧾 Πρώτες γραμμές:\n", df.head())
print("\nℹ️ Πληροφορίες:\n")
print(df.info())
print("\n🕳️ Null values:\n", df.isnull().sum())

# -----------------------------------------
# 🧼 Βήμα 4: Καθαρισμός + Target μετατροπή
# -----------------------------------------
def categorize(score):
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Medium"
    else:
        return "High"

df["Addicted_Level"] = df["Addicted_Score"].apply(categorize)

if "Student_ID" in df.columns:
    df.drop("Student_ID", axis=1, inplace=True)

print("\n📊 Κατανομή Κατηγοριών:\n", df["Addicted_Level"].value_counts())

# -----------------------------------------
# 🔠 Βήμα 5: Κωδικοποίηση Κατηγορικών
# -----------------------------------------
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# -----------------------------------------
# 🎯 Βήμα 6: Επιλογή Features για εκπαίδευση
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
# 🧪 Βήμα 7: Train/Test split + SMOTE
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n🔁 Κατανομή μετά το SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# -----------------------------------------
# 🤖 Βήμα 8: Εκπαίδευση Μοντέλου
# -----------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# -----------------------------------------
# 📊 Βήμα 9: Αξιολόγηση
# -----------------------------------------
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("🔵 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
