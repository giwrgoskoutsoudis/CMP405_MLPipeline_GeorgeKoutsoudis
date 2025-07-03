# 🧠 Machine Learning – Social Media Addiction Prediction

## 📌 Περιγραφή Dataset

Το dataset "Students Social Media Addiction" περιλαμβάνει δεδομένα από φοιτητές σχετικά με:

- Χρόνο χρήσης social media
- Ψυχική υγεία
- Ύπνο
- Ακαδημαϊκές επιδόσεις
- Διαπροσωπικές σχέσεις

Προέρχεται από την πλατφόρμα [Kaggle](https://www.kaggle.com/datasets) και είναι κατάλληλο για πρόβλημα ταξινόμησης (classification).

## 🔍 Περιγραφή Μοντέλου

Χρησιμοποιήσαμε τον αλγόριθμο **Random Forest Classifier** για να προβλέψουμε το επίπεδο εθισμού στα social media (Low, Medium, High) με βάση τα χαρακτηριστικά του κάθε φοιτητή.

- ✅ Accuracy: >99%
- 🔢 Training/Test split: 80/20
- 🧪 Μετρικές: Accuracy, Precision, Recall

## 🧪 Προετοιμασία & Εκπαίδευση

1. **Καθαρισμός Δεδομένων:** Αφαίρεση `Student_ID`, μετατροπή κατηγορικών χαρακτηριστικών με Label Encoding.
2. **Target Variable:** Δημιουργήθηκε νέα στήλη `Addicted_Level` από το `Addicted_Score`.
3. **Εκπαίδευση:** Το μοντέλο εκπαιδεύτηκε με τα υπόλοιπα χαρακτηριστικά για ταξινόμηση σε 3 επίπεδα εθισμού.

## 📂 Περιεχόμενα

- `ML_Model_SocialMedia_Addiction.ipynb`: Google Colab notebook με τον κώδικα του μοντέλου.
- `Students Social Media Addiction.csv`: Dataset ανεβασμένο μέσω Colab.
- `README.md`: Αυτό το αρχείο.
