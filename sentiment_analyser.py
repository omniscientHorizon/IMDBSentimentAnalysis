import pandas as pd
import nltk
import ssl
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

# load dataset
dataset_path = "IMDBDataset.csv"  
df = pd.read_csv(dataset_path)

# map sentiment to binary values
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# test on a smaller version first
# sse only the first 1000 rows for testing
# df = df.head(1000).copy()

# display dataset info and first few rows
print("Dataset Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# load stopwords and initialize WordNet Lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# function to get POS tags for Lemmatizer
def get_wordnet_pos(word):
    """Maps POS tags to WordNet tags for accurate lemmatization"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  

# function to preprocess & clean text
def preprocess_text(text):
    """
    Cleans the text by:
    - Removing HTML tags
    - Removing special characters & numbers
    - Converting to lowercase
    - Tokenizing
    - Removing stopwords
    - Lemmatizing using WordNet
    """
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^a-zA-Z]', ' ', text)  
    text = text.lower()  
    words = word_tokenize(text)  
    words = [word for word in words if word not in stop_words] 
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]  
    return ' '.join(words)

# preprocess the dataset
print("\nPreprocessing reviews...")
df['cleaned_review'] = df['review'].apply(preprocess_text)

# display first few rows of the cleaned dataset
print("\nFirst 5 Rows After Preprocessing:")
print(df[['review', 'cleaned_review']].head())

# convert text into TF-IDF features
print("\nConverting text to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000)  
X_tfidf = vectorizer.fit_transform(df['cleaned_review'])

print("TF-IDF Matrix Shape:", X_tfidf.shape)

# define features and target variable
X = X_tfidf
y = df['sentiment']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# get the number of rows in the sparse matrix
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# train a logistic regression model
print("\nTraining logistic regression model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# save trained model and vectorizer
print("\nSaving model and vectorizer...")
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully!")

# load the model and vectorizer
print("\nLoading saved model and vectorizer...")
loaded_model = joblib.load("sentiment_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# predict sentiment for a new review
new_review = ["This movie was absolutely fantastic!"]
new_review_tfidf = loaded_vectorizer.transform(new_review)
prediction = loaded_model.predict(new_review_tfidf)

# show the result
print("\nSentiment Prediction for New Review:")
print("Review:", new_review[0])
print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")