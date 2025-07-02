# 🧠 Maybank Marathon Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![NLP](https://img.shields.io/badge/Topic-Sentiment%20Analysis-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Twitter%20%26%20YouTube-informational)
![Model](https://img.shields.io/badge/Model-Machine%20Learning-lightgrey)

---

## 📌 Project Overview

**Maybank-Sentiment** is a social media sentiment analysis project built to evaluate the public perception of the **Maybank Marathon 2024** event using machine learning techniques. This end-to-end pipeline processes user-generated content from platforms such as **Twitter and YouTube**, extracts sentiment-labeled text, and generates insights through analytics and visualization.

This project was developed as a part of a professional consulting engagement with **PT Maybank Indonesia Tbk** to help improve brand reputation, public communication strategy, and future event planning.

---

## 🎯 Objectives

- Collect and analyze public opinion related to **Maybank Marathon 2024**
- Perform **automated sentiment classification** (positive, neutral, negative)
- Generate dashboards for **strategic insights**
- Provide a **scalable NLP pipeline** for future event monitoring

---

## 🔍 Data Sources

- **Twitter API** (via `snscrape`)
- **YouTube Comments** (via `googleapiclient`)
- Keywords used: `Maybank Marathon`, `#MaybankMarathon`, `#MaybankMarathon2024`, etc.

---

## 🧪 Features & Workflow

### 🗃️ 1. Data Collection
- Twitter scraping using `snscrape`
- YouTube video comment fetching via YouTube Data API
- Combined into unified Pandas DataFrame

### 🧹 2. Preprocessing
- Text cleaning (remove links, mentions, hashtags, emojis)
- Tokenization & normalization
- Stopword removal
- Lemmatization (NLTK + spaCy)

### 🧠 3. Sentiment Modeling
- Model used: **Logistic Regression / Naive Bayes**
- Vectorization: **TF-IDF**
- Sentiment labels: `positive`, `neutral`, `negative`
- Optionally supports external tools (TextBlob/VADER)

### 📈 4. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score
- Visualization:
  - Wordclouds per sentiment
  - Pie chart of sentiment distribution
  - Timeline plot of sentiment over time

### 📊 5. Dashboard/Reporting
- Generated insights prepared for stakeholder presentation
- Exported `.csv` and `.png` for visual charts

---

## 🧰 Tech Stack

| Layer        | Tools & Libraries                              |
|--------------|------------------------------------------------|
| Programming  | Python 3.11                                    |
| Data         | Pandas, NumPy                                  |
| NLP          | NLTK, spaCy, Sastrawi                          |
| ML Modeling  | scikit-learn, TF-IDF                           |
| Data Viz     | Matplotlib, Seaborn, Wordcloud                 |
| Scraping     | snscrape, google-api-python-client             |
| Deployment   | Jupyter Notebook, CSV exports                  |

---

## 🧬 Folder Structure

```text
Maybank-Sentiment/
│
├── data/ # Raw & processed dataset
├── models/ # Serialized models (optional)
├── notebooks/ # Jupyter notebooks for each stage
├── utils/ # Custom preprocessing modules
├── output/ # Charts & evaluation reports
│
├── collect_twitter.py # Scrape tweets
├── collect_youtube.py # Scrape YouTube comments
├── preprocess.py # Text cleaner and processor
├── train_model.py # Training pipeline
├── analyze_sentiment.py # Run predictions on new data
├── visualize.py # Generate charts
```

---

## 🧪 Sample Results

### 📊 Sentiment Distribution
![Sentiment Pie Chart](output/sentiment_pie.png)

### 🌐 Word Cloud (Positive Sentiment)
![Wordcloud Positive](output/wordcloud_positive.png)

---

## 📈 Model Performance

| Metric     | Value    |
|------------|----------|
| Accuracy   | 87.3%    |
| Precision  | 85.6%    |
| Recall     | 84.9%    |
| F1-score   | 85.2%    |

> *Model trained using 80-20 split and 10-fold cross-validation.*

---

## 🚀 How to Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/Aryasuta17/Maybank-Sentiment.git
cd Maybank-Sentiment
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run Script
```bash
python collect_twitter.py
python preprocess.py
python train_model.py
python visualize.py
```

## 📌 Use Case Scenarios
- Corporate brand monitoring
- Event-based public sentiment tracking
- Early warning system for negative publicity
- Enhancing PR & communications strategy

## 👤 Author 
#### Aryasuta

🔗 GitHub: @Aryasuta17

📫 Reach out for collaborations, portfolio, or AI research projects.

#### “Sentiment is not just opinion , it's the heartbeat of perception.”
