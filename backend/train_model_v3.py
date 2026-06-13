import pandas as pd
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import pickle


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


frames = []

# --- 1. Original US dataset (Fake.csv / True.csv) ---
fake_us = pd.read_csv("datasets_extracted/Fake.csv")
true_us = pd.read_csv("datasets_extracted/True.csv")
fake_us["content"] = (fake_us["title"].fillna("") + " " + fake_us["text"].fillna(""))
true_us["content"] = (true_us["title"].fillna("") + " " + true_us["text"].fillna(""))
fake_us["label"] = 0
true_us["label"] = 1
frames.append(fake_us[["content", "label"]])
frames.append(true_us[["content", "label"]])

# --- 2. IFND (Indian Fake News Dataset) ---
ifnd = pd.read_csv("/mnt/user-data/uploads/IFND.csv", encoding="latin1")
ifnd["content"] = ifnd["Statement"].fillna("")
ifnd["label"] = ifnd["Label"].apply(lambda x: 1 if str(x).strip().upper() == "TRUE" else 0)
frames.append(ifnd[["content", "label"]])

# --- 3. news_dataset.csv (Indian REAL/FAKE) ---
news_in = pd.read_csv("/mnt/user-data/uploads/news_dataset.csv")
news_in["content"] = news_in["text"].fillna("")
news_in["label"] = news_in["label"].apply(lambda x: 1 if str(x).strip().upper() == "REAL" else 0)
frames.append(news_in[["content", "label"]])

# --- 4. BharatFakeNewsKosh (English-translated statements + body) ---
bharat = pd.read_excel("/mnt/user-data/uploads/bharatfakenewskosh__3_.xlsx")
bharat_eng = bharat[bharat["Eng_Trans_Statement"].notna()].copy()
bharat_eng["content"] = (
    bharat_eng["Eng_Trans_Statement"].fillna("") + " " +
    bharat_eng["Eng_Trans_News_Body"].fillna("")
)
bharat_eng["label"] = bharat_eng["Label"].apply(lambda x: 1 if str(x).strip().upper() == "TRUE" else 0)
frames.append(bharat_eng[["content", "label"]])

# --- Combine all ---
df = pd.concat(frames, ignore_index=True)
df = df[df["content"].str.strip().str.len() > 5]
df["content"] = df["content"].apply(clean_text)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Combined dataset shape:", df.shape)
print(df["label"].value_counts())

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "Passive Aggressive": CalibratedClassifierCV(PassiveAggressiveClassifier(max_iter=1000, random_state=42)),
    "Linear SVC": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=3, scoring="accuracy")
    print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"{name} - Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

    if test_acc > best_score:
        best_score = test_acc
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} with test accuracy {best_score:.4f}")

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Best model and vectorizer saved.")

test_text = "Jaishankar speaks to Rubio, lodges strong protest over U.S. Navy attacks that killed three Indians"
cleaned = clean_text(test_text)
vec = vectorizer.transform([cleaned])
pred = best_model.predict(vec)[0]
proba = best_model.predict_proba(vec)[0]
print("\nSanity check (Jaishankar headline):")
print("Prediction:", "REAL" if pred == 1 else "FAKE", "| Proba:", proba)
