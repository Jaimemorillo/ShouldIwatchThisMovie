import sys
import numpy as np
sys.path.append('app')

from flask import Flask, render_template, request, abort, redirect, url_for

from controller.random_controller import DBRandomController
from controller.billboard_controller import DBBillboardController
data_path = 'data/'
models_path = 'models/'

# db_ctrl = DBRandomController()
db_random_ctrl = DBRandomController(data_path, models_path)
db_bb_ctrl = DBBillboardController(db_random_ctrl.mod, data_path, models_path)

app = Flask(__name__)


def render_carrousel(controller):
    if request.method == 'GET':
        return render_template('home.html',
                               ids=controller.sample.index.values, movies=controller.sample)

    elif request.method == 'POST':
        controller.update_sample()
        return render_template('home.html',
                               ids=controller.sample.index.values, movies=controller.sample)


def render_movie_page(id, controller):
    if request.method == 'GET':
        id = int(id)
        if controller.sample.title.get(id) is not None:
            return render_template('movie.html',
                                   id=id, movies=controller.sample)
        else:
            abort(404)

    elif request.method == 'POST':
        # set like of movie
        id = int(id)

        def set_like(key, page):
            value = request.form[key]
            value = int(value)

            if controller.sample.loc[id, 'like'] == value:
                controller.sample.loc[id, 'like'] = np.nan
            else:
                controller.sample.loc[id, 'like'] = value

            print(controller.sample.like)
            return redirect(page)

        if request.form.get("like") is not None:
            if 'random' in request.path:
                return set_like('like', '/random')
            else:
                return set_like('like', '/nowplaying')
        else:
            return set_like('like2', url_for(request.endpoint, id=str(id)))

# Home page
@app.route('/')
@app.route('/random',  methods=['GET', 'POST'])
def random():
    return render_carrousel(db_random_ctrl)

# Movie page
@app.route('/random/movie/<id>', methods=['GET', 'POST'])
def random_movie(id):
    return render_movie_page(id, db_random_ctrl)


@app.route('/nowplaying',  methods=['GET', 'POST'])
def nowplaying():
    return render_carrousel(db_bb_ctrl)

# Movie page
@app.route('/nowplaying/movie/<id>', methods=['GET', 'POST'])
def nowplaying_movie(id):
    return render_movie_page(id, db_bb_ctrl)


if __name__ == '__main__':
    app.run(debug=True)
