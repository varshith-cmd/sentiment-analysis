import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("sentiment_dataset.csv")

data.dropna(subset=['text', 'label'], inplace=True)

data['label'] = data['label'].map({'positive': 1, 'negative': 0, 'neutral': 2})

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

new_text = ["Amazing build quality!", "Not worth it at all", "It's just okay"]
new_vector = vectorizer.transform(new_text)
predictions = model.predict(new_vector)

print(predictions)
