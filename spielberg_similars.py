# James Keenan
# INST414
# Medium Post 3

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies_df = pd.read_csv('imdb_top_1000.csv', usecols=['Series_Title', 'IMDB_Rating', 'Meta_score', 'Director', 'Star1', 'Star2', 'Genre', 'Released_Year'])
movies_df = movies_df.replace([np.inf, -np.inf], 0).fillna("")
movies_df["id"] = movies_df.index

movies_df["similar_features"] = (
    str(movies_df["IMDB_Rating"]) 
    + " " + str(movies_df["Meta_score"]) 
    + " " + movies_df["Director"]
    + " " + movies_df["Star1"]
    + " " + movies_df["Star2"]
    + " " + movies_df["Genre"]
    + " " + str(movies_df["Released_Year"])
)

vectors = TfidfVectorizer()
feature_matrix = vectors.fit_transform(movies_df["similar_features"])

sim_matrix = cosine_similarity(feature_matrix)

def get_similar_movies(title, df, sim_matrix, top_n):
    title_row = df[(df['Series_Title'] == title)]
    title_index = title_row.index[0]

    sim_scores = list(enumerate(sim_matrix[title_index]))
    sorted_titles = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    results=[]
    for i, score in sorted_titles:
        results.append((df.iloc[i]['Series_Title'], score))
    return results

spielberg_movies = ["Jurassic Park", "Jaws", "Saving Private Ryan"]

for movie in spielberg_movies:
    print(f"Top 10 movies similar to '{movie}':")
    sim_movies = get_similar_movies(movie, movies_df, sim_matrix, 10)

    if isinstance(sim_movies, str):
        print(sim_movies)
    else:
        for title, score in sim_movies:
            print(f"{title} (Similarity: {score:.4f})")