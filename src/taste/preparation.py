import pandas as pd
import time

import tmdbsimple as tmdb
from requests.exceptions import HTTPError


class Preparation:

    tmdb.API_KEY = '38dd5c6c01713ef99903275d51e2fd68'

    def read_csv(self, path):

        data = pd.read_csv(path, sep='#', lineterminator='\n', encoding='utf-8')

        return data

    def get_overview(self, over_path):

        over = self.read_csv(over_path)

        return over

    def get_personal_taste(self, taste_path):

        taste = self.read_csv(taste_path)

        # Clean csv
        taste = taste[~taste['id'].str.contains('/')]
        taste['id'] = taste['id'].astype(int)

        taste = taste.dropna(subset=['like'])
        taste['like'] = taste['like'].astype(int)

        return taste

    def get_credits(self, credits_path):

        movie_credits = self.read_csv(credits_path)

        movie_credits['movie_id'] = movie_credits['movie_id'].astype(int)

        return movie_credits

    def merge_over_taste_credits(self, over, taste, movie_credits):

        data = taste.merge(over[['id', 'overview']], left_on='id', right_on='id')
        data = data.merge(movie_credits[['movie_id', 'cast', 'crew']], left_on='id', right_on='movie_id')
        data.drop(['movie_id'], axis=1, inplace=True)

        # Clean  empty
        data = data[~pd.isna(data.overview)]
        data['like'] = data['like'].astype(int)
        data.reset_index(inplace=True, drop=True)

        return data

    def api_request_by__id(self, movie_id):

        movie = tmdb.Movies(movie_id)

        def movie_request(movie):

            movie.info(language="es-ES")
            cast = movie.credits()['cast']
            crew = movie.credits()['crew']

            return movie.id, movie.original_title, movie.title, movie.overview, movie.genres, cast, crew, movie.release_date

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