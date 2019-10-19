import sys
import numpy as np
sys.path.append('app')
sys.path.append('../src')

from flask import Flask, render_template, url_for, request, abort, redirect

from controller.controller import DBController
path = '../data/'

db_ctrl = DBController()

app = Flask(__name__)

# Home page
@app.route('/')
@app.route('/home',  methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html',
                               ids=db_ctrl.sample.index.values, movies=db_ctrl.sample)

    elif request.method == 'POST':
        db_ctrl.update_sample()
        return render_template('home.html',
                               ids=db_ctrl.sample.index.values, movies=db_ctrl.sample)


# Movie page
@app.route('/movie/<id>', methods=['GET', 'POST'])
def movie(id):
    if request.method == 'GET':
        id = int(id)
        if db_ctrl.sample.title.get(id) is not None:
            return render_template('movie.html',
                                   id=id, movies=db_ctrl.sample)
        else:
            abort(404)

    elif request.method == 'POST':
        # set like of movie
        id = int(id)

        def set_like(key, page):
            value = request.form[key]
            value = int(value)

            if db_ctrl.sample.loc[id, 'like'] == value:
                db_ctrl.sample.loc[id, 'like'] = np.nan
            else:
                db_ctrl.sample.loc[id, 'like'] = value

            print(db_ctrl.sample.like)
            return redirect(page)

        if request.form.get("like") is not None:
            return set_like('like', '/home')

        else:
            return set_like('like2', '/movie/' + str(id))


if __name__ == '__main__':
    app.run(debug=True)
