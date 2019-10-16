import sys
sys.path.append('app')
sys.path.append('../src')

from flask import Flask, render_template, url_for, request, abort

from controller.controller import DBController
path = '../data/tmdb_spanish_overview.csv'

db_ctrl = DBController(path)
ids, movies = db_ctrl.get_4_random_movies()

app = Flask(__name__)

# Home page
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', ids=ids, movies=movies)


# Movie page
@app.route('/movie/<id>')
def movie(id):
    id = int(id)
    if movies.title.get(id) is not None:
        return render_template('movie.html', id=id, movies=movies)
    else:
        abort(404)

# Give like
@app.route('/movie/<id>/<l>', methods=['POST'])
def like(id, l):
    id = int(id)
    l = int(l)

    if l in [0, 1]:
        movies.taste[id] = l

    return None


if __name__ == '__main__':
    app.run(debug=True)
