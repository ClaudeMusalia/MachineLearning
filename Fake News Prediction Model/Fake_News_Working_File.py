# About the data set
# 1. id: unique id for a news article
# 2. title: the title of the news article
# 3. author: author of the news article
# 4. test: the text of the article; could be incomplete
# 5. label: a label that marks weather the news article is real of fake

# 1: Fake news
# 0: Real news

# import the dependencies
import nltk
import pandas as pd  # for making the dataFrames and storing the data in the dataframes
import numpy as np  # for making numpy arrays
import re  # known as regular expression, very useful in searching the text in a document
from nltk.corpus import stopwords  # NLTK stands for natural language tool kit. stop words that don't add value in
# articles, this function will remove such words
from nltk.stem.porter import PorterStemmer  # stemming takes a word and removes the suffix and prefix and returns the
# root of the word
from sklearn.feature_extraction.text import TfidfVectorizer  # used to convert text into feature vectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Printing the stopwords in english
'''nltk.download('stopwords')
print(stopwords.words('english'))'''

# Data Pre processing
# loading the dataset to a pandas dataFrame
news_dataset = pd.read_csv('train.csv')
'''print(news_dataset.shape)'''

# print the first five rows of the dataframe
'''print(news_dataset.head())'''

# counting the number of missing values in the dataset
'''print(news_dataset.isnull().sum())'''

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# for our prediction we are going to use title and author, thus we will combine the two. We leave the text data for
# now because it is so large. The title and author are usually very accurate for prediction.

# merging the author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
'''print(news_dataset['content'])'''

# separating the data and the label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# stemming procedure. The process of reducing a word to its root word. removing the suffix and prefixes.
# example: actor, actress, acting --> act
# we need to reduce the number of words to ensure that our model is accurate as possible.

port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  # regular expression (re) is useful for searching a paragraph
    # for a text. sub means substituting certain values. We are excluding all text that is not present in the set a-z
    # and A-Z. This will remove all the numbers and punctuation marks. all removed will be replaced by a place.
    stemmed_content = stemmed_content.lower()  # convert all text to lower case.
    stemmed_content = stemmed_content.split()  # split and converted to a lit
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    # performing stemming by removing stopwords that we saw earlier. so the function will choose words that are not in
    # stopwords.words('english')
    stemmed_content = ' '.join(stemmed_content)  # joining all the words
    return stemmed_content


news_dataset['content'] = news_dataset['content'].apply(stemming)

# separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# this is the data that we will feed our machine learning model.
# we need to convert the text data into numerical data through the vectorizer function
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)  # will convert all the values into feature vectors.

# splitting the dataset to training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the model. the logistic regression model.
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluating the model
# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# Making a predictive system
X_new = X_test[0]

prediction = model.predict(X_new)
print(prediction)

if prediction[0] == 0:
    print('The news is Real')
else:
    print('The News is fake')


# for binary classification problems, logistic regression is the best model to use. 
