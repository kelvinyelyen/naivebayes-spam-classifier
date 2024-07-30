import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data into features (X) and target (y)
X = df['message']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the vectorizer
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# Define prediction function
def predict_spam_or_ham(text):
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)
    return 'Spam' if prediction[0] == 1 else 'Not Spam (ham)'

# Streamlit app
st.title('Spam Detection')
st.write("Enter a message below to classify it as spam or not spam (ham):")

# Text input for prediction
user_input = st.text_area("Message")

if st.button('Classify'):
    if user_input:
        result = predict_spam_or_ham(user_input)
        st.write(f'The message is classified as: **{result}**')
    else:
        st.write('Please enter a message for classification.')

