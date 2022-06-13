import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import string
import pickle

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


def load_data(database_filepath):
    """
    Function: load data from database and return X and y.
    Args:
      database_filepath: database file name included path
    Return:
      X: messages for X
      y: labels part in messages for y
      category_names: category names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disasterTab',engine) 
    X = df.message
    Y = df.loc[:, 'related':'direct_report']    
    category_names=Y.columns.values   
    
    return X, Y, category_names    


def tokenize(text):
    """
    Function: tokenize the text
    Args:  source string
    Return:
    clean_tokens: cleaned string list
    """
    table = text.maketrans(dict.fromkeys(string.punctuation))
    words = word_tokenize(text.lower().strip().translate(table))
    words = [word for word in words if word not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in words]
    lemmed = [WordNetLemmatizer().lemmatize(word, pos='v') for word in lemmed]
    stemmed = [PorterStemmer().stem(word) for word in lemmed]
    clean_tokens = stemmed

    return clean_tokens


def build_model():
    """
    Function: build model that consist of pipeline
    Return:
      cv: Grid Search model
    """

    pipeline  = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
            
    parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'clf__estimator__min_samples_split': [2, 4]
    }
 
    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    return cv #pipeline  


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: evaluate model 
    Args:
      model,
      X_test: X test dataset
      Y_test: y test dataset
      category_names: category names of y
    """

    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print("classification report for " + col,
              '\n', classification_report(Y_test.values[:,i],y_pred[:,i]))

def save_model(model, model_filepath):
    """
    Function: save model as pickle file.
    Args:
      model: target model
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #print(category_names)
        
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
