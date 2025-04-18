import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load features
X, y = joblib.load("video_features.pkl")

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save model
joblib.dump(clf, "modules/video_classifier.pkl")
print("✅ Classifier saved to modules/video_classifier.pkl")

# Report
preds = clf.predict(X)
print(classification_report(y, preds, target_names=["Real", "Fake"]))
