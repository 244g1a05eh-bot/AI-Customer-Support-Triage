import streamlit as st
import joblib
import csv
import pandas as pd

# Load saved models
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
category_model = joblib.load("models/category_model.pkl")
urgency_model = joblib.load("models/urgency_model.pkl")

# Page title
st.title("🤖 AI Customer Support Ticket Triage")

st.write("Enter a customer support ticket below.")

# Ticket input
ticket = st.text_area("Customer Ticket")

# Predict button
if st.button("Analyze Ticket"):

    if ticket.strip() == "":
        st.warning("Please enter a customer ticket.")

    else:
        # Convert ticket into TF-IDF numbers
        ticket_tfidf = vectorizer.transform([ticket])

        # Predict category
        category = category_model.predict(ticket_tfidf)[0]

        # Get category confidence
        category_confidence = max(
            category_model.predict_proba(ticket_tfidf)[0]
        )

        # Predict urgency
        urgency = urgency_model.predict(ticket_tfidf)[0]

        # Get urgency confidence
        urgency_confidence = max(
            urgency_model.predict_proba(ticket_tfidf)[0]
        )

        # Overall confidence
        confidence = min(
            category_confidence,
            urgency_confidence
        )

        # Display results
        st.subheader("AI Prediction")

        st.write("Category:", category)
        st.write("Urgency:", urgency)
        st.write("Confidence:", f"{confidence * 100:.2f}%")

        # Human review
        if confidence < 0.60:

            st.warning(
                "⚠️ Low confidence - Human review required."
            )

            with open(
                "human_review.csv",
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    ticket,
                    category,
                    urgency,
                    f"{confidence * 100:.2f}%"
                ])

        else:

            st.success(
                "✅ Prediction confidence is acceptable."
            )

# Human Review Records
st.subheader("👤 Human Review Records")

try:

    review_data = pd.read_csv("human_review.csv")

    st.dataframe(review_data)

except FileNotFoundError:

    st.info("No human review records yet.")