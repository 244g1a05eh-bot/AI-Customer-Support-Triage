# AI-Customer-Support-Triage

AI-based customer support ticket classification and urgency prediction.

## Project Description

This project uses Machine Learning to automatically analyze customer support tickets.

It predicts:

- Ticket category
- Ticket urgency
- Prediction confidence

Low-confidence predictions are sent for human review.

### Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Streamlit

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

```bash
python train.py
```

### 3. Run the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

## Project Structure

```text
AI-Customer-Support-Triage/
├── data/
│   └── tickets.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── category_model.pkl
│   └── urgency_model.pkl
├── app.py
├── train.py
├── requirements.txt
└── human_review.csv
```

## Output

The application displays:

- Predicted Category
- Predicted Urgency
- Confidence Score

Low-confidence predictions are recorded for human review.
