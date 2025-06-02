from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the data
movies = pickle.load(open('movies.pkl', 'rb'))
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

# Reset index for easy lookup
movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title'])

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

@app.route('/')
def home():
    return render_template('home.html', movie_list=movies['title'].values)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form['movie']
    recommendations = get_recommendations(movie)
    return render_template('recommend.html', movie=movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
