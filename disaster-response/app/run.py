import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disasterTab', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # data of first visualization
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # data for second visualization    
    labels=df.iloc[:,4:].sum().sort_values(ascending=False).reset_index()
    labels.columns=['category','count']
    label_values=labels['count'].values.tolist()
    label_names=labels['category'].values.tolist()

    # get top 10 categories
    categories = df.iloc[:,4:]
    cat_mean = categories.mean().sort_values(ascending=False)[1:11]
    cat_names = list(cat_mean.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_values,
                    marker = dict(color='green')
                )
            ],

            'layout': {
                'title': "Chart Frequency of Categories of Messages",
                'yaxis': {
                    'title':"Message Category Frequency"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_mean
                )
            ],

            'layout': {
                'title': 'Top 10',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
   ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()