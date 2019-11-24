import pandas as pd
import time

import tmdbsimple as tmdb
from requests.exceptions import HTTPError


class Preparation:

    tmdb.API_KEY = '38dd5c6c01713ef99903275d51e2fd68'

    def read_csv(self, path):

        data = pd.read_csv(path, sep='#', lineterminator='\n', encoding='utf-8')

        # Clean ids
        data.dropna(subset=['id'], inplace=True)
        data['id'] = data['id'].astype(str)
        data = data[~data['id'].str.contains('/')]
        data['id'] = data['id'].astype(int)

        data.set_index('id', inplace=True)

        data['id'] = data.index

        return data

    def get_overview(self, over_path):

        over = self.read_csv(over_path)
        over.dropna(subset=['title', 'overview'], inplace=True)

        return over

    def get_personal_like(self, like_path):

        like = self.read_csv(like_path)

        like = like.dropna(subset=['like'])
        like['like'] = like['like'].astype(int)
        like = like[['id', 'like']]

        return like

    def merge_over_like(self, over, like):

        data = over.drop(['like'], axis=1).merge(like.drop(['id'], axis=1),
                                                 left_index=True, right_index=True)

        # Clean  empty
        data = data[~pd.isna(data.overview)]

        return data

    @staticmethod
    def api_request_by__id(movie_id):

        movie = tmdb.Movies(movie_id)

        def movie_request(movie):

            movie.info(language="es-ES")
            cast = movie.credits()['cast']
            crew = movie.credits()['crew']

            return movie.id, movie.original_title, movie.title, movie.overview, \
                movie.genres, cast, crew, movie.release_date

        try:

            return movie_request(movie)

        except HTTPError as err:

            # print(err.response.status_code)
            if err.response.status_code == 429:

                time.sleep(10)
                try:
                    return movie_request(movie)
                except:
                    return movie_id, 'original_title', 'title', 'overview', 'genres', 'cast', 'crew', 'release_date'

            else:

                return movie_id, 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'

        except:

            print('Fallo en: ' + str(movie_id))

    @staticmethod
    def api_request_now_playing():

        movies = tmdb.Movies("now_playing")
        response = movies.info(language='es-ES')
        results = response['results']
        ids = [i['id'] for i in results]
        casts = []
        crews = []

        for idx in ids:
            movie = tmdb.Movies(idx)
            cast = movie.credits()['cast']
            crew = movie.credits()['crew']

            casts = casts + [cast]
            crews = crews + [crew]

        data = {
            'id': ids,
            'original_title': [i['original_title'] for i in results],
            'title': [i['title'] for i in results],
            'overview': [i['overview'] for i in results],
            'genres': [i['genre_ids'] for i in results],
            'cast': casts,
            'crew': crews,
            'release_date': [i['release_date'] for i in results]
        }

        now_playing = pd.DataFrame(data=data)
        now_playing['id'] = now_playing['id'].astype(int)

        now_playing.set_index('id', inplace=True)

        now_playing['id'] = now_playing.index

        return now_playing


