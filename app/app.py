from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

films = {
	0: {
		'title': 'It Capítulo 2',
		'prediction': '70'
	},
	1: {
		'title': 'Origen',
		'prediction': '80'
	},
	2: {
		'title': 'Vengadores',
		'prediction': '90'
	},
	3: {
		'title': 'Joker',
		'prediction': '50'
	}
}


@app.route('/')
@app.route('/home')
def home(film="it"):
	return render_template('index.html', films=films)


if __name__ == '__main__':
	app.run(debug=True)
