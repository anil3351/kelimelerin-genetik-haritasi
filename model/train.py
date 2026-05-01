import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import joblib

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "dataset.csv")
VECT_PATH = os.path.join(HERE, "vectorizer.pkl")
CLF_PATH = os.path.join(HERE, "classifier.pkl")
LBL_PATH = os.path.join(HERE, "label_encoder.pkl")

def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")

    df["word"] = df["word"].astype(str).str.strip().str.lower()
    df["origin"] = df["origin"].astype(str).str.strip()
    df["notes"] = df["notes"].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["word", "origin"])
    df = df[df["word"] != ""]
    df = df.drop_duplicates(subset=["word"], keep="first")

    return df

def train_and_save():
    df = load_dataset()

    X = df["word"]
    y = df["origin"]

    vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    Xv = vect.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = LogisticRegression(max_iter=3000, solver="liblinear")
    clf.fit(Xv, y_enc)

    try:
        scores = cross_val_score(clf, Xv, y_enc, cv=5, scoring="accuracy")
        print("5-fold doğruluk skorları:", scores)
        print("Ortalama doğruluk:", scores.mean())
    except Exception as e:
        print("Cross-validation yapılamadı:", e)

    joblib.dump(vect, VECT_PATH)
    joblib.dump(clf, CLF_PATH)
    joblib.dump(le, LBL_PATH)

    print("Eğitim tamamlandı.")
    print("-", VECT_PATH)
    print("-", CLF_PATH)
    print("-", LBL_PATH)

if __name__ == "__main__":
    train_and_save()