import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import numpy as np
from skmultilearn.model_selection import IterativeStratification


class Processing:

    def clean_overviews(self, df):

        df = df.dropna(subset=['overview']).reset_index(drop=True)

        df = df[~df['overview'].str.contains('Recuerda que puedes ver esta película')]
        df = df[df.overview.str.split().apply(len) > 10].reset_index(drop=True)

        df = df[['overview', 'genres']]

        return df

    def get_overviews_describe(self, df):

        df['overview'].str.split().apply(len).describe().astype(int)

    def clean_genres(self, df):

        def eval_cell(cell):

            try:

                cell_array = eval(cell)

            except:

                cell_array = []

            return cell_array

        def get_genres(cell):

            cell_array = eval_cell(cell)

            if len(cell_array) > 0:
                ids_list = sorted([v['name'] for v in cell_array])

            else:
                ids_list = []

            return ids_list

        # Crate Dataframe with ids
        def create_df_genres(ids, column_name):

            enc = MultiLabelBinarizer()
            np_ids = enc.fit_transform(ids)

            # Save encoder in a pickle

            column_names = []
            print('Num of ' + column_name + ': ' + str(len(enc.classes_)))

            for c in enc.classes_:
                column_names.append(str(c))

            df_ids = pd.DataFrame(data=np_ids, index=df.index, columns=column_names)

            return df_ids

        # Merge dataframe ids with data
        def merge_ids(data, column_name):

            ids = data[column_name].apply(lambda x: get_genres(x))

            df_ids = create_df_genres(ids, column_name)

            new_df = data.copy()
            new_df = new_df.join(df_ids)

            return new_df

        df = merge_ids(df, 'genres')
        df['genres'] = df['genres'].apply(lambda x: get_genres(x)).apply(tuple)
        df = df.drop(['Película de TV'], axis=1)

        # Elimina las peliculas sin genero
        df['n_genres'] = df.iloc[:, -18:].sum(axis=1)
        df = df[df['n_genres'] != 0].reset_index(drop=True)
        df = df.iloc[:, :-1]

        return df

    def get_genres_describe(self, df):

        sum_genres = df.iloc[:, -18:].sum(axis=0).astype(int).sort_values().to_dict()

        vc = df['genres'].value_counts()
        vc = vc.describe().astype(int)

        return sum_genres, vc

    def cut_sentence(self, df, max_seq_length, train):

        # Create datasets (Only take up to max_seq_length words for memory)
        X = df['overview'].tolist()
        X = [' '.join(t.split()[0:max_seq_length]) for t in X]
        X = np.array(X, dtype=object)[:, np.newaxis]

        if train:
            y = df.iloc[:, -18:].values
            return X, y

        return X

    def iterative_train_test_split(self, X, y, test_size):

        stratifier = IterativeStratification(n_splits=2, order=4,
                                             sample_distribution_per_fold=[test_size, 1.0 - test_size])
        train_indexes, test_indexes = next(stratifier.split(X, y))

        X_train, y_train = X[train_indexes], y[train_indexes, :]
        X_test, y_test = X[test_indexes], y[test_indexes, :]

        return X_train, X_test, y_train, y_test
