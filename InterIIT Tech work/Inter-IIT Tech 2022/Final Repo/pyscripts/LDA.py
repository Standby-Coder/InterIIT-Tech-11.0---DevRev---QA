import json
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk

# Load the SQuAD dataset
# with open('squad_dataset.json', 'r') as f:
#     squad_data = json.load(f)
nltk.download('stopwords')
squad_data = pd.read_csv('train_data.csv')
paragraphs_data = pd.read_csv("train_input_paragraph.csv")  
# Preprocess the text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
texts = []

for paragraph in squad_data['Paragraph']:
    for qa in paragraph['qas']:
        question = qa['question'].lower()
        answer = qa['answers'][0]['text'].lower()
        question_tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(question) if token not in stop_words]
        answer_tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(answer) if token not in stop_words]
        texts.append(question_tokens + answer_tokens)

# Create a dictionary from the texts
dictionary = corpora.Dictionary(texts)

# Create a corpus from the texts
corpus = [dictionary.doc2bow(text) for text in texts]

# Train the LDA model
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)

# Create a function to get the most likely topic for a new question
def get_topic(question):
    question_tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(question.lower()) if token not in stop_words]
    question_bow = dictionary.doc2bow(question_tokens)
    topic_probs = ldamodel[question_bow]
    return max(topic_probs, key=lambda x: x[1])[0]

# Create a function to get the most likely answer for a question
def get_answer(question):
    question_topic = get_topic(question)
    topic_answers = [text for text, topic in zip(texts, ldamodel[corpus]) if topic[0][0] == question_topic]
    return max(topic_answers, key=lambda x: len(x))

# Test the model with a sample question
for i in range(10):
    question = squad_data['question'][i]
    answer = squad_data['answer'][i]
    print("Question:", question)
    print("Answer:", answer)
    print("Predicted Answer:", get_answer(question))