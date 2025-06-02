import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the dataset
movies = pd.read_csv('movies.csv', encoding='ISO-8859-1')
movies = movies.head(1000)  # Make sure 'movies.csv' is in the same folder

# Fill NaN with empty string
movies['genres'] = movies['genres'].fillna('')

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Compute the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the cosine similarity matrix and movie data
pickle.dump(cosine_sim, open('cosine_sim.pkl', 'wb'))
pickle.dump(movies, open('movies.pkl', 'wb'))
