import sys
import os
sys.path.extend('../src')

from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from taste.preparation import Preparation
from movies.movie import Movie

pre = Preparation()
overs = pre.get_overview('../data/tmdb_spanish_overview.csv')

movies = overs[0:4]

app = Flask(__name__)

movies = {
	0: {
		'title': movies['title'][0],
		'prediction': '70'
	},
	1: {
		'title': movies['title'][1],
		'prediction': '80'
	},
	2: {
		'title': movies['title'][2],
		'prediction': '90'
	},
	3: {
		'title': movies['title'][3],
		'prediction': '50'
	}
}


@app.route('/')
@app.route('/home')
def home(film="it"):
	return render_template('index.html', movies=movies)


if __name__ == '__main__':
	app.run(debug=True)
