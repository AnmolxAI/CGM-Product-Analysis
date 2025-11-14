# CGM-Product-Analysis

Here’s a clean, professional **README.md** for your project, based on the notebook contents you uploaded.

---

# **NLP Analysis of Consumer Posts on Diabetes CGM Devices**

## **📌 Project Overview**

This project analyzes **consumer sentiment and experiences** related to **Continuous Glucose Monitoring (CGM)** devices using real-world posts from platforms such as Twitter, Facebook, Reddit, and online forums.
The objective is to understand user pain points, common themes, frequent challenges, product perceptions, and general consumer sentiment toward CGM technologies.

---

## **📁 Repository Structure**

```
├── data/
│   ├── raw/                 # Raw dataset(s)
│   ├── processed/           # Cleaned or intermediate outputs
│
├── notebooks/
│   ├── Team5_NLP_Analysis_of_DCGM_Consumer_Posts.ipynb
│
├── src/
│   ├── preprocessing.py     # Data cleaning utilities
│   ├── text_processing.py   # NLP preprocessing functions
│   ├── analysis.py          # Topic modelling and sentiment analysis
│   ├── visualization.py     # Word clouds, plots, etc.
│
├── README.md
└── requirements.txt
```

---

## **🔍 Workflow Summary**

### **1. Data Loading & Exploration**

* Inspect raw data columns
* Understand structure, dataset size, missing values
* Explore consumer post distributions

---

### **2. Data Cleaning**

Includes extensive preprocessing steps:

#### a. Remove duplicate rows

#### b. Handle missing values

#### c. Drop irrelevant columns

#### d. Merge post text + title context

#### e. Convert text to lowercase

#### f. Filter out users with large follower counts (to reduce influencer bias)

#### g. Remove posts from professionals, news sources, and non-consumer accounts

---

### **3. Text Preprocessing**

Steps used to prepare data for NLP:

* Remove URLs, newline characters
* Handle emojis and Unicode entities
* Check and remove hashtags, mentions, and retweet markers
* Remove punctuation
* Tokenize text
* Remove stopwords
* Extract and remove common bigrams/trigrams
* Lemmatization (if applied in notebook)

---

### **4. Exploratory NLP Analysis**

* Frequency analysis of words, bigrams, and trigrams
* Word clouds representing consumer concerns
* Key phrase extraction
* Distribution of post topics

---

### **5. Sentiment Analysis**

* Apply rule-based or model-based sentiment scoring
* Compare sentiment across platforms or user groups
* Identify positive/negative themes

---

### **6. Topic Modeling**

(if present in later cells)

* LDA / NMF topic modeling
* Interpretation of dominant themes
* Visualization of topic-word distributions

---

## **📊 Visualizations**

Typical outputs include:

* Word clouds
* Bar charts of frequent words/phrases
* Sentiment distribution plots
* Topic model visualizations

---

## **🛠️ Technologies Used**

* Python
* Pandas
* NLTK
* SpaCy
* WordCloud
* Matplotlib / Seaborn
* Scikit-Learn

---

## **▶️ How to Run the Notebook**

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:

   ```bash
   jupyter notebook notebooks/Team5_NLP_Analysis_of_DCGM_Consumer_Posts.ipynb
   ```
4. Run cells sequentially.

---

## **🎯 Project Goals**

* Identify dominant consumer pain points
* Understand how users perceive CGM devices
* Detect misinformation or recurring misconceptions
* Support CGM product development, UX improvements, and patient education strategies

---

## **👥 Contributors**

* Anmol Kumar
* Mihiko Jo
* Xiaotong Yang
* Wendy Guangli Liu

