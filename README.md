# 🧠 Machine Learning – Social Media Addiction Prediction

## 📌 Περιγραφή

Αυτό το notebook περιλαμβάνει την υλοποίηση ενός Machine Learning μοντέλου που προβλέπει το επίπεδο εθισμού ενός φοιτητή στα social media, βασισμένο σε χαρακτηριστικά όπως χρόνος χρήσης, ύπνος και επιπτώσεις στην καθημερινότητα.

Χρησιμοποιείται το dataset **"Students Social Media Addiction"** από την πλατφόρμα [Kaggle](https://www.kaggle.com/datasets), κατάλληλο για πρόβλημα ταξινόμησης.

## 👤 Ονοματεπώνυμο & ΑΜ

**Ονοματεπώνυμο:** Γιώργος Κουτσούδης  
**Αριθμός Μητρώου (ΑΜ):** 12345678

## 🔍 Περιγραφή Μοντέλου

Χρησιμοποιήσαμε τον αλγόριθμο **Random Forest Classifier** για να προβλέψουμε το επίπεδο εθισμού στα social media:  
- `Low` – Χαμηλό  
- `Medium` – Μέτριο  
- `High` – Υψηλό

**Βασικά Χαρακτηριστικά:**
- Accuracy: >99%
- Train/Test split: 80/20
- Μετρικές: Accuracy, Precision, Recall
- Χρήση SMOTE για ισορροπία μεταξύ κατηγοριών

## 🧪 Βήματα Εκπαίδευσης

1. **Καθαρισμός Δεδομένων:** Αφαίρεση `Student_ID`, Label Encoding για κατηγορικά.
2. **Target Variable:** Δημιουργήθηκε στήλη `Addicted_Level` από το `Addicted_Score`.
3. **Εκπαίδευση Μοντέλου:** Με Random Forest Classifier και SMOTE.
4. **Αξιολόγηση:** Με Accuracy Score, Classification Report, Confusion Matrix.

## ▶️ Οδηγίες Εκτέλεσης

1. **Ανοίξτε το αρχείο:**  
   `ML_Model_SocialMedia_Addiction.ipynb` μέσω Google Colab ή Jupyter Notebook.

2. **Βήμα προς βήμα εκτέλεση:**  
   Εκτελέστε κάθε κελί διαδοχικά για να:
   - Εισάγετε τα δεδομένα
   - Καθαρίσετε το dataset
   - Εκπαιδεύσετε το μοντέλο
   - Δείτε τα αποτελέσματα αξιολόγησης

3. **Προϋποθέσεις:**  
   Απαιτούνται οι παρακάτω βιβλιοθήκες:
   - `pandas`, `numpy`
   - `sklearn`
   - `matplotlib`, `seaborn`
   - `imblearn`

   Αν εργάζεστε σε τοπικό περιβάλλον, μπορείτε να εγκαταστήσετε τις βιβλιοθήκες με:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
