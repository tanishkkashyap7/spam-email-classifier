import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import pickle

nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()

    words = text.split()

    y = []

    for i in words:
        if i.isalnum():
            y.append(i)

    y = [word for word in y if word not in stopwords.words('english')]

    y = [ps.stem(word) for word in y]

    return " ".join(y)

df = pd.read_csv('data/spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]

df.rename(columns={
    'v1': 'label',
    'v2': 'message'
}, inplace=True)

df['label'] = df['label'].map({'ham':0, 'spam':1})

df['transformed_text'] = df['message'].apply(transform_text)

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text']).toarray()

y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("Model Saved!")