import pandas as pd


class Read:

    def __init__(self):
        pass

    @staticmethod
    def read_from_csv(path, sep, encoding):

        data = pd.read_csv(path=path, sep=sep, encoding=encoding)

        return data
