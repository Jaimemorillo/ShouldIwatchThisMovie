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
    return render_template('home.html', movies=movies)


@app.route('/movie/<id>')
def movie(id):
    try:
        title, overview, prediction = movies.get_by_id(id)
        print(title)
        return render_template('movie.html', title=title, overview=overview, prediction=prediction)
    except:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True)
