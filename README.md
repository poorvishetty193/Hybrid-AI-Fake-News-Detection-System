# 🧠 Hybrid AI-Based Fake News Detection System

## 📌 Overview

This project implements a **Hybrid AI-Based Fake News Detection System** using both **Machine Learning (ML)** and **Deep Learning (DL)** techniques. The system classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) and neural network models.

The project combines:
- Baseline ML Model (Logistic Regression)
- Deep Learning Model (LSTM Neural Network)
- Confidence-based prediction
- Interactive Streamlit Dashboard
- Model comparison visualization

This system demonstrates the practical application of Artificial Intelligence in detecting misinformation.

---

# 🏗 System Architecture
```
User Input
↓
Text Cleaning (NLP)
↓
Feature Extraction
↓
┌───────────────────────────┐
│ Logistic Regression (ML) │
└───────────────────────────┘
↓
┌───────────────────────────┐
│ LSTM Deep Learning Model │
└───────────────────────────┘
↓
Confidence Score
↓
Visualization Dashboard
```
---

# 🤖 Where AI is Implemented

Artificial Intelligence is implemented through:

## 1️⃣ Deep Learning (LSTM Neural Network)

- Embedding Layer for semantic representation
- LSTM (Long Short-Term Memory) network for sequence learning
- Automatic feature extraction
- Backpropagation-based optimization
- Non-linear pattern learning

Unlike traditional ML, the LSTM model:
- Learns contextual relationships
- Understands word sequences
- Captures semantic meaning

This qualifies the project as an **AI-based system**, not just basic ML.

---

# 📂 Project Structure
```
FakeNewsAI/
│
├── train_models.py # Trains ML and DL models
├── app.py # Streamlit dashboard
├── utils.py # Text cleaning helper functions
├── requirements.txt # Required dependencies
├── Fake.csv # Fake news dataset
├── True.csv # Real news dataset
├── lr_model.pkl # Saved Logistic Regression model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── dl_model.h5 # Saved LSTM deep learning model
└── tokenizer.pkl # Saved tokenizer
```
---

# 🛠 Environment Requirements

## ✅ Python Version
Python 3.10.x (Recommended)

⚠ TensorFlow does NOT support Python 3.13/3.14

Check version:
```
python --version
```
OR
```
py -3.10 --version
```
---

# 📦 Required Libraries
```
pandas
numpy
scikit-learn
tensorflow
streamlit
nltk
joblib
matplotlib
```

Install using:
```
py -3.10 -m pip install -r requirements.txt
```
---

# 🚀 Steps to Run the Project

## Step 1️⃣ Clone Repository
```
git clone https://github.com/poorvishetty193/Hybrid-AI-Fake-News-Detection-System.git
cd FakeNewsAI
```
---

## Step 2️⃣ Install Dependencies
```
py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install -r requirements.txt
```
---

## Step 3️⃣ Train Models
```
py -3.10 train_models.py
```
This will:
- Train Logistic Regression model
- Train LSTM deep learning model
- Save models locally

---

## Step 4️⃣ Run Streamlit Dashboard
```
py -3.10 -m streamlit run app.py
```

Open the browser link displayed in terminal.

---
# 📊 Models Used

## 🔹 1. Logistic Regression (Baseline ML Model)

- Uses TF-IDF feature extraction
- Acts as performance comparison baseline
- Fast and lightweight

## 🔹 2. LSTM Deep Learning Model (AI Model)

- Embedding Layer (Word representation)
- LSTM layer (Sequence learning)
- Dropout (Regularization)
- Dense output layer (Binary classification)

### Why LSTM?
- Captures contextual relationships
- Understands word order
- Learns semantic dependencies

---

# 📈 Features Implemented

✔ Hybrid ML + Deep Learning Architecture  
✔ Text Preprocessing using NLP  
✔ Tokenization and Sequence Padding  
✔ TF-IDF Vectorization  
✔ Confidence Score Prediction  
✔ Model Comparison  
✔ Visualization Dashboard  
✔ Probability Graph  
✔ Interactive Web UI  

---

# 🧪 Model Evaluation

The system evaluates:

- Accuracy Score
- Train-Test Split validation
- Confidence probability
- Model comparison (ML vs DL)

Expected Performance:
- Logistic Regression: ~92%
- LSTM Model: ~95-97%

---

# 🎨 Dashboard Features

The Streamlit dashboard provides:

- Text input interface
- Deep Learning prediction result
- Confidence score display
- ML baseline comparison
- Probability visualization chart

---

 # 🏗 ScreenShot

<img width="1920" height="821" alt="Screenshot (467)" src="https://github.com/user-attachments/assets/834e7da7-f478-4694-a07b-3781575bf93b" />

---

# 📚 References

- Kaggle Fake News Dataset
- TensorFlow Documentation
- Scikit-learn Documentation
- NLP Research Papers on Fake News Detection

---

# 🏆 Conclusion

This project demonstrates a real-world application of Artificial Intelligence by combining:

- Traditional Machine Learning
- Deep Learning (LSTM)
- Natural Language Processing
- Interactive Deployment

The hybrid architecture improves misinformation detection accuracy and provides a scalable solution for digital content verification.

---

## 👨‍💻 Author

Poorvi Shetty

---

⭐ If you found this project useful, consider giving it a star!
