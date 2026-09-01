# Task 4: Email Spam Detection with Machine Learning

**Track:** Data Science  
**Internship:** OASIS INFOBYTE  
**Theme:** Crystal Intelligence

---

## Live Demo Video

Watch the complete project walkthrough and demo here:

**[Click to Watch Live Demo](https://drive.google.com/file/d/1rPlGqSSWDo6XCvQMARRi9RiKbKts7Cps/view?usp=sharing)**

---

## Objective
Build an NLP binary classifier that distinguishes spam emails from legitimate (ham) emails using text preprocessing, TF-IDF feature extraction, and machine learning algorithms.

---

## Tech Stack
- **Python** - pandas, numpy for data manipulation
- **NLTK** - tokenization, stemming, stopword removal
- **scikit-learn** - TF-IDF, Naive Bayes, Logistic Regression, SVM
- **matplotlib/seaborn** - EDA visualizations
- **WordCloud** - text pattern visualization
- **Jupyter Notebook** - interactive development

---

## Dataset
**SMS Spam Collection Dataset** (UCI Machine Learning Repository)
- 5,574 messages total
- ~87% Ham (legitimate) | ~13% Spam (unsolicited)
- Slightly imbalanced - recall optimization is critical

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Class distribution analysis
- Message length statistics
- Word count comparisons

### 2. Text Preprocessing Pipeline
```
Lowercase -> Remove Punctuation -> Remove Numbers -> Tokenize -> Remove Stopwords -> Stem
```

### 3. TF-IDF Vectorization
- Unigrams + Bigrams (ngram_range = 1,2)
- Max 3,000 features
- Term Frequency-Inverse Document Frequency

### 4. Model Training (80/20 Stratified Split)
| Model | Type | Strength |
|-------|------|----------|
| Multinomial Naive Bayes | Probabilistic | Fast, works well with text |
| Logistic Regression | Linear | Interpretable, high performance |
| Support Vector Machine | Maximum Margin | Robust, good generalization |

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curves

### 6. WordCloud Visualization
- Spam trigger words: "free", "win", "urgent", "call now", "prize"
- Ham patterns: conversational, personal terms

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 97.8% | 98.2% | 96.5% | 97.3% |
| **Logistic Regression** | **98.5%** | **99.1%** | **97.2%** | **98.1%** |
| SVM | 98.2% | 98.8% | 96.8% | 97.8% |

### Best Model: Logistic Regression
- Highest F1-Score (0.981) - optimal precision-recall balance
- Interpretable coefficients show which words drive spam classification
- Fast training and inference - ideal for production email filters

---

## Why Recall Matters for Spam Detection

In spam detection, a **False Negative** (spam classified as ham) is far more dangerous than a **False Positive** (ham in spam folder).

| Scenario | Risk Level |
|----------|-----------|
| False Positive | Minor inconvenience - user checks spam folder |
| False Negative | **CRITICAL** - phishing, malware, fraud reach inbox |

Missing a phishing email can lead to:
- Identity theft
- Financial loss
- Corporate data breaches
- Malware infections

**We optimize for Recall > 95%** while maintaining reasonable precision.

---

## Files

| File | Description |
|------|-------------|
| `index.html` | Crystal-themed project showcase website (52KB) |
| `email_spam_detection.py` | Complete Python ML pipeline |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download the SMS Spam Collection Dataset from:
- [Kaggle - SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Or [UCI Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

Place the CSV file in the same directory.

### 3. Run Python Script
```bash
python email_spam_detection.py
```

### 4. View Website
Open `index.html` in any modern web browser.

---

## Website Features

The `index.html` showcases the project with a **Crystal Intelligence** theme:

- **Custom Crystal Cursor** - Glowing ring + trail particles
- **3D Crystal Prism** - Rotating triangular prism (CSS 3D)
- **Crystal Grid Floor** - 3D perspective grid with glow pulse
- **80 Floating Particles** - Crystal dust, diamond shards, sparkles
- **8 Floating Crystal Shards** - Hexagons & diamonds with complex float
- **7 Light Beams** - Vertical gradient beams with staggered pulse
- **Caustics Overlay** - Underwater light refraction effect
- **Ice Crack Texture** - Subtle diagonal line pattern
- **Hover Burst Particles** - 12 particles explode on card hover
- **Parallax Mouse Layers** - 3 depth-responsive background layers
- **Text Shimmer** - Gradient position animation on titles
- **GSAP Scroll Animations** - Staggered reveals, score ring animations
- **Live Spam Scanner** - Real-time heuristic-based demo

---

## Author
[Your Name]  
Data Science Intern | OASIS INFOBYTE

---

## License
This project is for educational purposes as part of the OASIS INFOBYTE internship program.
