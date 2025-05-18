import sys
import joblib

def extract_features(url):
    return {
        "url_length": len(url),
        "has_https": int(url.startswith("https")),
        "has_at_symbol": int("@" in url),
        "has_hyphen": int("-" in url),
        "num_dots": url.count("."),
        "contains_ip": int(any(char.isdigit() for char in url.split("/")[2] if url.startswith("http")) if "://" in url else 0),
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python classifier.py <url>")
        return
    
    url = sys.argv[1]

    # Load the trained model
    model = joblib.load("phishing_model.pkl")

    # Extract features from input URL
    features = extract_features(url)

    # Convert features dict to list of values in the same order used in training
    feature_order = ["url_length", "has_https", "has_at_symbol", "has_hyphen", "num_dots", "contains_ip"]
    feature_vector = [features[feat] for feat in feature_order]

    # Predict using the model
    pred = model.predict([feature_vector])[0]

    if pred == 1:
        print(f"The URL '{url}' is predicted to be PHISHING.")
    else:
        print(f"The URL '{url}' is predicted to be LEGITIMATE.")

if __name__ == "__main__":
    main()
