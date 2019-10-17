import sys
import numpy as np
sys.path.append('app')
sys.path.append('../src')

from flask import Flask, render_template, url_for, request, abort, redirect

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
@app.route('/movie/<id>', methods=['GET', 'POST'])
def movie(id):
    if request.method == 'GET':
        id = int(id)
        if movies.title.get(id) is not None:
            return render_template('movie.html', id=id, movies=movies)
        else:
            abort(404)

    if request.method == 'POST':
        # set taste of movie
        id = int(id)

        def set_taste(key, page):
            value = request.form[key]
            value = int(value)

            if movies['taste'][id] == value:
                movies['taste'][id] = np.nan
            else:
                movies['taste'][id] = value

            print(movies.taste)
            return redirect(page)

        if request.form.get("taste") is not None:
            return set_taste('taste', '/home')

        else:
            return set_taste('taste2', '/movie/' + str(id))


if __name__ == '__main__':
    app.run(debug=True)
