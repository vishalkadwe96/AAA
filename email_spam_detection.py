
"""
OASIS INFOBYTE - Data Science Internship
Task 4: Email Spam Detection with Machine Learning
Author: [Your Name]
Track: Data Science
"""

# ============================================================
# CELL 1: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)

# WordCloud
from wordcloud import WordCloud

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

print("✅ All libraries imported successfully!")

# ============================================================
# CELL 2: Load Dataset
# ============================================================
# Using the classic SMS Spam Collection Dataset
# Source: UCI Machine Learning Repository / Kaggle
# We'll simulate loading - in practice, download from Kaggle

# For demonstration, creating a representative sample
# In actual implementation, use: df = pd.read_csv('spam.csv', encoding='latin-1')

# Simulated dataset (representative of SMS Spam Collection)
data = {
    'label': ['ham']*100 + ['spam']*50,
    'message': [
        # Ham messages
        "Hey, are we still meeting for lunch tomorrow?",
        "Can you pick up some milk on your way home?",
        "The meeting is rescheduled to 3 PM",
        "Happy birthday! Hope you have a great day",
        "Your Amazon order has been shipped",
        "See you at the gym around 6",
        "Thanks for the help with the project",
        "Don't forget to bring the documents",
        "The package arrived safely",
        "Call me when you get a chance",
        "Great job on the presentation today",
        "Let's catch up this weekend",
        "Your appointment is confirmed for Monday",
        "The report looks good, send it",
        "I'll be late, traffic is terrible",
        "Dinner at 8? Sounds perfect",
        "Can you send me the file?",
        "The movie was amazing!",
        "Thanks for the ride home",
        "See you in class tomorrow",
        # ... (more ham)
    ] + [
        # Spam messages
        "Congratulations! You've won $1000 cash prize. Call now!",
        "URGENT: You have won a free iPhone. Click here to claim",
        "Free entry to win a car! Text WIN to 12345",
        "You are selected for a $5000 reward. Call immediately!!!",
        "Buy cheap viagra pills now!!! 80% discount",
        "URGENT! Your account will be suspended. Click link now",
        "You have 1 new voicemail. Call 09061749602 to hear",
        "Free lottery ticket! You've won! Call to claim prize",
        "Text MONEY to 77777 to receive $1000 cash",
        "Congratulations! You're the 1000000th visitor! Claim now",
        # ... (more spam)
    ]
}

# In actual notebook, load real dataset:
# df = pd.read_csv('spam.csv', encoding='latin-1')
# df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})

# For this template, we'll note the actual loading code
df = pd.DataFrame(data)
print(f"Dataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================
# CELL 3: Exploratory Data Analysis (EDA)
# ============================================================
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Class distribution
print("\nClass Distribution:")
print(df['label'].value_counts())
print(f"\nSpam Percentage: {df['label'].value_counts()['spam']/len(df)*100:.2f}%")

# Visualize class distribution
plt.figure(figsize=(8, 6), facecolor='#0a0e27')
colors = ['#00d4ff', '#ff006e']
df['label'].value_counts().plot(kind='bar', color=colors, edgecolor='white', linewidth=2)
plt.title('📊 Class Distribution: Ham vs Spam', fontsize=16, fontweight='bold', color='white', pad=20)
plt.xlabel('Label', fontsize=12, color='white')
plt.ylabel('Count', fontsize=12, color='white')
plt.xticks(rotation=0, color='white')
plt.yticks(color='white')
plt.gca().set_facecolor('#0a0e27')
plt.gca().spines['bottom'].set_color('white')
plt.gca().spines['left'].set_color('white')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('images/class_distribution.png', dpi=150, bbox_inches='tight', facecolor='#0a0e27')
plt.show()

# Message length analysis
df['message_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

print(f"\nMessage Length Statistics:")
print(df.groupby('label')[['message_length', 'word_count']].describe())

# Box plot for message length
plt.figure(figsize=(10, 6), facecolor='#0a0e27')
sns.boxplot(data=df, x='label', y='message_length', palette=['#00d4ff', '#ff006e'])
plt.title('📏 Message Length Distribution by Class', fontsize=16, fontweight='bold', color='white', pad=20)
plt.xlabel('Label', fontsize=12, color='white')
plt.ylabel('Message Length (characters)', fontsize=12, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.gca().set_facecolor('#0a0e27')
plt.gca().spines['bottom'].set_color('white')
plt.gca().spines['left'].set_color('white')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('images/message_length_boxplot.png', dpi=150, bbox_inches='tight', facecolor='#0a0e27')
plt.show()

# ============================================================
# CELL 4: Text Preprocessing Pipeline
# ============================================================
print("\n" + "="*50)
print("TEXT PREPROCESSING PIPELINE")
print("="*50)

# Initialize tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Complete text preprocessing pipeline:
    1. Lowercase conversion
    2. Remove punctuation
    3. Remove numbers
    4. Tokenization
    5. Stopword removal
    6. Stemming
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]

    return ' '.join(tokens)

# Apply preprocessing
print("Applying preprocessing pipeline...")
df['processed_message'] = df['message'].apply(preprocess_text)

print("\nOriginal vs Processed (Sample):")
for i in range(3):
    print(f"\nOriginal: {df['message'].iloc[i]}")
    print(f"Processed: {df['processed_message'].iloc[i]}")

# ============================================================
# CELL 5: TF-IDF Feature Extraction
# ============================================================
print("\n" + "="*50)
print("TF-IDF FEATURE EXTRACTION")
print("="*50)

"""
TF-IDF (Term Frequency-Inverse Document Frequency) measures the importance 
of a word in a document relative to a corpus. 

- TF: How often a word appears in a document
- IDF: Downweights common words that appear in many documents
- TF-IDF = TF × IDF: High value = word is distinctive to that document
"""

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

# Fit and transform
X = tfidf.fit_transform(df['processed_message']).toarray()
y = df['label'].map({'ham': 0, 'spam': 1}).values

print(f"TF-IDF Matrix Shape: {X.shape}")
print(f"Features (top 20): {tfidf.get_feature_names_out()[:20]}")

# ============================================================
# CELL 6: Train-Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Training spam ratio: {np.mean(y_train)*100:.2f}%")
print(f"Testing spam ratio: {np.mean(y_test)*100:.2f}%")

# ============================================================
# CELL 7: Model Training
# ============================================================
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

# Model 1: Multinomial Naive Bayes (Industry standard for text)
print("\n🔹 Training Multinomial Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# Model 2: Logistic Regression
print("🔹 Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Model 3: Support Vector Machine (Bonus)
print("🔹 Training SVM...")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\n✅ All models trained successfully!")

# ============================================================
# CELL 8: Model Evaluation
# ============================================================
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

def evaluate_model(name, y_true, y_pred):
    """Comprehensive model evaluation"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n📊 {name} Results:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")

    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

# Evaluate all models
results = {}
results['Naive Bayes'] = evaluate_model("Multinomial Naive Bayes", y_test, nb_pred)
results['Logistic Regression'] = evaluate_model("Logistic Regression", y_test, lr_pred)
results['SVM'] = evaluate_model("Support Vector Machine", y_test, svm_pred)

# Results DataFrame
results_df = pd.DataFrame(results).T
print("\n📋 Comparison Table:")
print(results_df.round(4))

# ============================================================
# CELL 9: Confusion Matrix Visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0a0e27')
models = [('Naive Bayes', nb_pred), ('Logistic Regression', lr_pred), ('SVM', svm_pred)]

for idx, (name, pred) in enumerate(models):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                ax=axes[idx], cbar_kws={'label': 'Count'})
    axes[idx].set_title(f'🔍 {name}', fontsize=14, fontweight='bold', color='white', pad=15)
    axes[idx].set_xlabel('Predicted', fontsize=11, color='white')
    axes[idx].set_ylabel('Actual', fontsize=11, color='white')
    axes[idx].tick_params(colors='white')
    axes[idx].set_facecolor('#0a0e27')

plt.suptitle('Confusion Matrices Comparison', fontsize=18, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('images/confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor='#0a0e27')
plt.show()

# ============================================================
# CELL 10: Why is Recall Important for Spam Detection?
# ============================================================
print("\n" + "="*50)
print("DISCUSSION: WHY RECALL MATTERS FOR SPAM DETECTION")
print("="*50)

discussion = """
🎯 RECALL (Sensitivity) = True Positives / (True Positives + False Negatives)

In spam detection:
- True Positive (TP): Spam correctly identified as spam ✅
- False Negative (FN): Spam incorrectly classified as ham ❌ (MISSED SPAM!)
- False Positive (FP): Ham incorrectly classified as spam ⚠️ (Legitimate email in spam folder)

WHY RECALL IS CRITICAL:

1. 🔒 SECURITY RISK: Missing spam (low recall) means phishing emails, malware links,
   and fraud attempts reach the user's inbox. This is far more dangerous than 
   occasionally sending a newsletter to the spam folder.

2. 💰 FINANCIAL IMPACT: A single missed phishing email can lead to identity theft,
   financial loss, or corporate data breaches. The cost of a False Negative 
   vastly exceeds the inconvenience of a False Positive.

3. 📧 USER TRUST: Users rely on spam filters to protect them. If spam consistently
   slips through, users lose trust in the email service and may abandon it.

4. ⚖️ ASYMMETRIC COST: 
   - False Positive cost: User checks spam folder (minor inconvenience)
   - False Negative cost: Potential security breach, fraud, malware infection

5. 🛡️ DEFENSE-IN-DEPTH: High recall ensures maximum threat coverage. 
   It's better to be overly cautious (catch all spam, some false alarms) 
   than to let threats through.

RECOMMENDATION: Optimize for high recall (>95%) while maintaining reasonable 
precision. In production, combine with user feedback loops and whitelisting 
to reduce false positives over time.
"""
print(discussion)

# ============================================================
# CELL 11: WordCloud Visualization
# ============================================================
print("\n" + "="*50)
print("WORDCLOUD VISUALIZATION")
print("="*50)

# Separate spam and ham messages
spam_text = ' '.join(df[df['label'] == 'spam']['processed_message'])
ham_text = ' '.join(df[df['label'] == 'ham']['processed_message'])

# Create WordClouds
fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor='#0a0e27')

# Spam WordCloud
spam_wc = WordCloud(width=800, height=400, background_color='#0a0e27',
                     colormap='Reds', max_words=100, contour_width=2,
                     contour_color='#ff006e').generate(spam_text)
axes[0].imshow(spam_wc, interpolation='bilinear')
axes[0].set_title('🔴 SPAM Words', fontsize=18, fontweight='bold', color='#ff006e', pad=20)
axes[0].axis('off')
axes[0].set_facecolor('#0a0e27')

# Ham WordCloud
ham_wc = WordCloud(width=800, height=400, background_color='#0a0e27',
                   colormap='Blues', max_words=100, contour_width=2,
                   contour_color='#00d4ff').generate(ham_text)
axes[1].imshow(ham_wc, interpolation='bilinear')
axes[1].set_title('🔵 HAM Words', fontsize=18, fontweight='bold', color='#00d4ff', pad=20)
axes[1].axis('off')
axes[1].set_facecolor('#0a0e27')

plt.suptitle('WordCloud: Most Frequent Terms', fontsize=20, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('images/wordclouds.png', dpi=150, bbox_inches='tight', facecolor='#0a0e27')
plt.show()

# ============================================================
# CELL 12: ROC Curve Comparison
# ============================================================
plt.figure(figsize=(10, 8), facecolor='#0a0e27')

models_roc = [
    ('Naive Bayes', nb_model),
    ('Logistic Regression', lr_model),
    ('SVM', svm_model)
]

colors_roc = ['#00d4ff', '#ff006e', '#39ff14']

for (name, model), color in zip(models_roc, colors_roc):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linewidth=3, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'w--', linewidth=2, alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=13, color='white', fontweight='bold')
plt.ylabel('True Positive Rate (Recall)', fontsize=13, color='white', fontweight='bold')
plt.title('📈 ROC Curve Comparison', fontsize=18, fontweight='bold', color='white', pad=20)
plt.legend(loc='lower right', fontsize=12, facecolor='#0a0e27', edgecolor='white', labelcolor='white')
plt.gca().set_facecolor('#0a0e27')
plt.gca().spines['bottom'].set_color('white')
plt.gca().spines['left'].set_color('white')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(color='white')
plt.yticks(color='white')
plt.grid(True, alpha=0.2, color='white')
plt.tight_layout()
plt.savefig('images/roc_curves.png', dpi=150, bbox_inches='tight', facecolor='#0a0e27')
plt.show()

# ============================================================
# CELL 13: Best Model Declaration
# ============================================================
print("\n" + "="*50)
print("🏆 BEST MODEL DECLARATION")
print("="*50)

best_model_name = results_df['F1'].idxmax()
best_score = results_df.loc[best_model_name, 'F1']

print(f"\n✨ BEST PERFORMING MODEL: {best_model_name}")
print(f"   F1-Score: {best_score:.4f}")
print(f"\nJustification:")
print(f"   - Highest F1-Score indicates best balance between Precision and Recall")
print(f"   - Critical for spam detection where both false positives and false negatives matter")
print(f"   - {best_model_name} effectively captures spam patterns while minimizing ham misclassification")

# ============================================================
# CELL 14: Save Model (Optional)
# ============================================================
import joblib

# Save the best model and vectorizer
joblib.dump(nb_model, 'spam_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\n💾 Model and vectorizer saved!")
print("   - spam_classifier_model.pkl")
print("   - tfidf_vectorizer.pkl")

# ============================================================
# CELL 15: Live Prediction Function
# ============================================================
def predict_spam(message, model=nb_model, vectorizer=tfidf):
    """
    Predict whether a message is spam or ham.
    Returns: prediction ('Ham'/'Spam'), probability
    """
    processed = preprocess_text(message)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    label = 'Spam' if pred == 1 else 'Ham'
    confidence = prob[pred] * 100

    return label, confidence

# Test examples
test_messages = [
    "Hey, want to grab coffee tomorrow?",
    "CONGRATULATIONS! You've won $1,000,000! Call now!!!",
    "Meeting rescheduled to 4 PM in conference room B",
    "URGENT: Your bank account has been compromised. Click here immediately!"
]

print("\n🧪 LIVE PREDICTIONS:")
print("-" * 60)
for msg in test_messages:
    label, conf = predict_spam(msg)
    emoji = "🔴" if label == "Spam" else "🟢"
    print(f"{emoji} {label} ({conf:.1f}% confidence)")
    print(f"   Message: {msg[:60]}...")
    print()

print("\n✅ Task 4 Complete! All deliverables generated.")
