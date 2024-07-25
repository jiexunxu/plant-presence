import pandas as pd
import csv
import warnings
import pathlib
import pickle


class Glc23Parser:
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
    RAW_DATA_DIR = f'{BASE_DIR}/data/'
    PROCESSED_DATA_DIR = f'{BASE_DIR}/data/'

    def __init__(self, processed_data_filename):
        self.processed_data_filename = processed_data_filename

    def read_maxent_data(self, maxent_filename):
        df = pd.read_csv(f'{self.RAW_DATA_DIR}{maxent_filename}', sep=';')
        df = df.iloc[: , :-1]
        return df
    
    def fill_missing_column(self, df, column_index):
        mean_value = df.iloc[:, column_index].mean()
        df.iloc[:, column_index].fillna(mean_value, inplace=True)

    def load_presence_absence_data(self, filename, top_species_count):
        pa_data = pd.read_csv(filename, sep=';', usecols = ['patchID', 'speciesId', 'dayOfYear'])
        pa_data["region"] = pa_data["patchID"].map(str).map(str)
        pa_data['count'] = 1
        species_count = pa_data.groupby(by='speciesId').sum().sort_values(by='count', ascending=False).index.tolist()
        top_species_list = species_count[:top_species_count]
        top_species_list.sort()
        pa_data = pa_data.pivot_table('count', index='region', columns='speciesId').fillna(-1).astype(int)
        return [pa_data, top_species_list]

    def load_env_data(self, env_filename, env_offset):
        env_data = pd.read_csv(env_filename, sep=';')
        env_data["region"] = env_data["patchID"].map(str)
        one_hot_encoding = pd.get_dummies(env_data['landCov'], prefix='landCov')
        env_data = env_data.join(one_hot_encoding)
        env_data = env_data.drop('landCov', axis=1)
        env_data = pd.concat([env_data.iloc[:, env_offset:]], axis=1)
        regions = env_data['region']
        env_data.drop(labels=['region'], axis=1, inplace = True)
        env_data.insert(0, 'region', regions)
        for i in range(1, len(env_data.columns)):
            self._fill_missing_column(env_data, i)
        return env_data

    def load_1km_presence_data(self, filename, result_rows, top_species_list):
        presence_1km = pd.read_csv(filename, sep=';')
        presence_1km["region"] = presence_1km["patchID"].map(str)
        result = pd.DataFrame(0, index=result_rows, columns=top_species_list, dtype=int)
        with warnings.catch_warnings():
            for _, row in presence_1km.iterrows():
                speciesId = row['speciesId'] 
                if speciesId in result.columns:
                    result.loc[result.index == row['region'], row['speciesId']] = row['n']
        return result

    def load_maxent_train_data(self, filename, result_rows, top_species_list):
        result = pd.DataFrame(0, index=result_rows, columns=top_species_list, dtype=int)
        with open(filename, "r") as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader)
            for row in reader:
                species_list = row[2].split(' ')
                region = row[0] + ' ' + row[1]
                result.loc[result.index == region, :] = -1
                for species in species_list:
                    speciesId = int(species)
                    if speciesId in result.columns:
                        result.loc[result.index == region, speciesId] = 1
        return result

    def load_maxent_test_data(self, filename, result_rows, top_species_list):
        result = pd.DataFrame(-1, index=result_rows, columns=top_species_list, dtype=int)
        with open(filename, "r") as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            row_count = 0
            for row in reader:
                if row_count == 0:
                    row_count = row_count + 1
                    continue
                species_list = row[1].split(' ')
                for species in species_list:
                    speciesId = int(species)
                    if speciesId in result.columns:
                        result.iloc[row_count - 1, result.columns.get_loc(speciesId)] = 1
                row_count = row_count + 1
        return result
    
    def save_processed_data(self, data):
        filename = f'{self.PROCESSED_DATA_DIR}{self.processed_data_filename}'
        with open(filename, 'wb') as file:
            pickle.dump(data, file)