

class Movie:

    def __init__(self, id, original_title, title, overview, genres, cast=[], crew=[], release_date=''):

        self.id = id
        self.original_title = original_title
        self.title = title
        self.overview = overview
        self.genres = genres
        self.cast = cast
        self.crew = crew
        self.release_date = release_date


