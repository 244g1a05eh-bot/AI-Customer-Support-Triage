import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib
# Load dataset
data = pd.read_csv("data/tickets.csv")  

# Get ticket text
X = data["ticket_text"]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert text into numbers
X_tfidf = vectorizer.fit_transform(X)

print("Number of tickets:", len(X))
print("TF-IDF shape:", X_tfidf.shape)
from sklearn.linear_model import LogisticRegression

# Category is our target
y = data["category"]
# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)
# Create and train the category model
category_model = LogisticRegression()
category_model.fit(X_tfidf, y)

print("\nCategory model trained successfully!")
# Evaluate category model
category_test_prediction = category_model.predict(X_test)

print("Actual categories:", list(y_test))
print("Predicted categories:", list(category_test_prediction))
print("\nCategory Evaluation:")
# Create confusion matrix
cm = confusion_matrix(y_test, category_test_prediction)

print("\nConfusion Matrix:")
print(cm)
print(classification_report(y_test, category_test_prediction))
# Test with a new customer ticket
new_ticket = ["I was charged twice for my order"]

# Convert new ticket into TF-IDF numbers
new_ticket_tfidf = vectorizer.transform(new_ticket)

# Predict category
prediction = category_model.predict(new_ticket_tfidf)

print("\nNew ticket:", new_ticket[0])
print("Predicted category:", prediction[0])
# Urgency model
y_urgency = data["urgency"]

urgency_model = LogisticRegression()
urgency_model.fit(X_tfidf, y_urgency)

print("\nUrgency model trained successfully!")
# Predict urgency for the new ticket
urgency_prediction = urgency_model.predict(new_ticket_tfidf)

print("Predicted urgency:", urgency_prediction[0])
# Save trained models
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(category_model, "models/category_model.pkl")
joblib.dump(urgency_model, "models/urgency_model.pkl")