import sys
sys.path.append('app')
sys.path.append('../src')

from flask import Flask, render_template, url_for, request, abort

from controller.controller import Controller
path = '../data/tmdb_spanish_overview.csv'

ctrl = Controller(path)
movies = ctrl.get_4_random_movies()

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', ids=movies.ids, movies=movies)


@app.route('/movie/<id>')
def movie(id):
    try:
        return render_template('movie.html', id=int(id), movies=movies)
    except:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True)
