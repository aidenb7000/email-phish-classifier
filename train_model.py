import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Feature extraction function
def extract_features(url):
    return {
        "url_length": len(url),
        "has_https": int(url.startswith("https")),
        "has_at_symbol": int("@" in url),
        "has_hyphen": int("-" in url),
        "num_dots": url.count("."),
        "contains_ip": int(any(char.isdigit() for char in url.split("/")[2] if url.startswith("http")) if "://" in url else 0),
    }

# Load data
phishing_path = "data/phishing_urls.csv"
legit_path = "data/legitimate_urls.csv"

if not os.path.exists(phishing_path) or not os.path.exists(legit_path):
    print("Missing phishing_urls.csv or legitimate_urls.csv in /data")
    exit()

print("Loading URLs...")
phishing_urls = pd.read_csv(phishing_path, header=None, names=["url"])
legitimate_urls = pd.read_csv(legit_path, header=None, names=["url"])

phishing_urls["label"] = 1
legitimate_urls["label"] = 0

# Combine and shuffle
df = pd.concat([phishing_urls, legitimate_urls], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Extracting features...")
features = df["url"].apply(extract_features).apply(pd.Series)
X = features
y = df["label"]

# Train-test split
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, "phishing_model.pkl")
print("Model trained and saved to phishing_model.pkl")

