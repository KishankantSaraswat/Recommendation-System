import math
import os
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Helper functions
def prepare_data(x):
    return str(x).lower().replace(" ", "")

def create_soup(x):
    return f"{x['Genre']} {x['Tags']} {x['Actors']} {x['ViewerRating']}"

def get_recommendations(title, cosine_sim):
    title = title.replace(' ', '').lower()
    idx = indices.get(title, None)
    if idx is None:
        return pd.DataFrame()  # Return an empty DataFrame if the movie title is not found

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:51]
    movie_indices = [i[0] for i in sim_scores]
    result = netflix_data.iloc[movie_indices]
    result.reset_index(inplace=True)
    return result

# Data Loading and Preprocessing
netflix_data = pd.read_csv("Netflix.csv", encoding='latin-1', index_col='Title')
netflix_data.index = netflix_data.index.str.title()
netflix_data = netflix_data[~netflix_data.index.duplicated()]
netflix_data.rename(columns={'View Rating': 'ViewerRating'}, inplace=True)

Language = netflix_data['Languages'].str.get_dummies(',')
Lang = set(Language.columns.str.strip().values.tolist())
Titles = set(netflix_data.index.str.title().to_list())

# Convert appropriate columns to categorical to save memory
netflix_data['Genre'] = netflix_data['Genre'].astype('category')
netflix_data['Tags'] = netflix_data['Tags'].astype('category')
netflix_data['Actors'] = netflix_data['Actors'].astype('category')
netflix_data['ViewerRating'] = netflix_data['ViewerRating'].astype('category')

netflix_data['IMDb Score'] = netflix_data['IMDb Score'].fillna(6.6)

# Create 'soup' feature for all movies
new_features = ['Genre', 'Tags', 'Actors', 'ViewerRating']
for feature in new_features:
    netflix_data[feature] = netflix_data[feature].map(prepare_data)

netflix_data['soup'] = netflix_data.apply(create_soup, axis=1)

# Create the count matrix and cosine similarity matrix
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(netflix_data['soup'])

# Using a sparse representation of the cosine similarity matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix, dense_output=False)

# Reset index and create a Series for title indices
netflix_data.reset_index(inplace=True)
indices = pd.Series(netflix_data.index, index=netflix_data['Title'].str.lower().str.replace(" ", ""))

@app.route('/')
def index():
    return render_template('index.html', languages=Lang, titles=Titles)

@app.route('/about', methods=['POST'])
def getvalue():
    global df
    movienames = request.form.getlist('titles')
    languages = request.form.getlist('languages')
    df = pd.DataFrame()

    for moviename in movienames:
        result = get_recommendations(moviename, cosine_sim2)
        for language in languages:
            language_filtered = result[result['Languages'].str.contains(language, case=False)]
            df = pd.concat([df, language_filtered], ignore_index=True)

    df.drop_duplicates(keep='first', inplace=True)
    df.sort_values(by='IMDb Score', ascending=False, inplace=True)

    images = df['Image'].tolist()
    titles = df['Title'].tolist()
    return render_template('result.html', titles=titles, images=images)

@app.route('/moviepage/<name>')
def movie_details(name):
    global df
    details_list = df[df['Title'] == name].to_numpy().tolist()
    return render_template('moviepage.html', details=details_list[0])

if __name__ == '__main__':
    app.run(debug=True)
