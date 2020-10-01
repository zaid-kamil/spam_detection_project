import streamlit as st
import pickle
import nltk 
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def cleanupText(message):
    message =  message.translate(str.maketrans('','',string.punctuation)) # remove basic puncutation
    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words('english')]
    return " ".join(words)

def load_model(path ='models/clf.pk'):
    with open(path,'rb') as f:
        return pickle.load(f)

st.title('Message Spam detection')

with st.spinner('loading AI model'):
    model = load_model()
    vectorizer =  load_model('models/tfidfvec.pk')
    st.success("models loaded into memory")

message = st.text_area('enter your sms text',value='hi there')
btn = st.button('submit to analyse')
if btn:
    stemmer = SnowballStemmer('english')
    clean_msg = cleanupText(message)
    data = vectorizer.transform([clean_msg])
    data = data.toarray()
    prediction = model.predict(data)
    st.title("our prediction")
    if prediction[0] == 0:
        st.header('Normal message')
    elif prediction[0] == 1:
        st.header("Spam message") 
    else:
        st.error("something fishy")