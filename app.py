# Main script file that predicts

# importing dependencies
import streamlit as st          # framework
import pickle           # to open model
import nltk         # nltk library for preprocessing
from nltk.corpus import stopwords           # for removing stopwords
import string           # to remove strings
from nltk.stem import PorterStemmer         # for stemming

# to interact with the operating system's file system
import os

# get the parent directory path
parent_directory = os.getcwd()

# paths
model_path = os.path.join(parent_directory, 'model') 


# """ 4 things to do: """
# 1. preprocess
# 2. vectorize
# 3. predict
# 4. display


def transform_text(text):
    """ Function to perform data preprocessing """

    # lower the text
    text = text.lower()

    # word tokenization
    text = nltk.word_tokenize(text)

    # fetching alpha numeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # clear y
    text = y[:]
    y.clear()

    # fetching words that are not stopwords and that are not punctuations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # clear y
    text = y[:]
    y.clear()

    # stemming
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    # converting list to string
    return " ".join(y)


# open vectorizer and model
tfidf = pickle.load(open(os.path.join(model_path, 'vectorizer.pkl'),'rb'))
model = pickle.load(open(os.path.join(model_path, 'model.pkl'),'rb'))


#----- frontend part -----#

# side bar
st.sidebar.info("Welcome to the SMS Spam Classfier dashboard!")
st.title('SMS Spam Classifier')         # display title
st.markdown("---")
input_sms = st.text_area('Enter the message')            # display single-line text input widget.

# adding a button
if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown("This message is **Spam**")
    else:
        st.markdown("This message is **Not Spam**")


# Define usage instructions as a string
usage_instructions = """
**User Instructions**

1. Type a message to check for spam.
2. Click on **Predict** button.

"""

spam_messages_example = """
1. You are awarded a SiPix Digital Camera! call 09061221061 from landline.
2. You have an important customer service announcement. Call FREEPHONE 0800 542 0825 now!

"""

with st.sidebar:
    st.markdown("---")
    st.markdown(usage_instructions)
    st.markdown("---")
    st.info('Some examples of spam messages: ')
    st.info(spam_messages_example)


with st.container():
    st.markdown("---")
    st.subheader("About the Dashboard")
    st.markdown("Welcome to the SMS Spam Classifier Dashboard!")
    st.markdown("This interactive application uses Naive Bayes algorithm and predicts whether a sms is spam or not.")
    st.markdown("---")
    st.subheader("Contact Information")
    st.markdown("Feel free to reach out to me if you have any questions or feedback. You can find me on:")
    st.markdown("Mail: [amanbhatt.1997.ab@gmail.com](mailto:amanbhatt.1997.ab@gmail.com)")
    st.markdown("Linkedin: [amanbhatt97](https://www.linkedin.com/in/amanbhatt1997/)")
    st.markdown("Github: [amanbhatt97](https://github.com/amanbhatt97)")
    st.markdown("Checkout my portfolio [here](https://amanbhatt97.github.io/portfolio/).")
    st.markdown("---")


