import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

nltk.download('stopwords')

# Load Data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true])
df = df[["text", "label"]]

# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------- Logistic Regression (Baseline ML) --------
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

lr_pred = lr_model.predict(X_test_tfidf)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)

joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# -------- LSTM Deep Learning Model --------
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=300)
X_test_pad = pad_sequences(X_test_seq, maxlen=300)

dl_model = Sequential()
dl_model.add(Embedding(10000, 128, input_length=300))
dl_model.add(LSTM(128))
dl_model.add(Dropout(0.5))
dl_model.add(Dense(1, activation='sigmoid'))

dl_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
dl_model.fit(X_train_pad, y_train, epochs=3, batch_size=64)

loss, dl_accuracy = dl_model.evaluate(X_test_pad, y_test)
print("LSTM Deep Learning Accuracy:", dl_accuracy)

dl_model.save("dl_model.h5")
joblib.dump(tokenizer, "tokenizer.pkl")