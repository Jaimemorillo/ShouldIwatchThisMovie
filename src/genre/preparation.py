import pandas as pd
import glob


class Preparation:


    def read_csv(self, path):

        data = pd.read_csv(path, sep='#', lineterminator='\n')

        return data

    def read_full_db(self, path):

        all_files = glob.glob(path + "/*.csv")
        print(len(all_files))

        li = []

        for filename in all_files:
            df = self.read_csv(filename)
            li.append(df)

        data = pd.concat(li, axis=0, ignore_index=True)
        data = data.sort_values('id').reset_index(drop=True)

        return data

