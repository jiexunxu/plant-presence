import random
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import pathlib


class GbifParser:
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
    RAW_DATA_DIR = f'{BASE_DIR}/data/'
    PROCESSED_DATA_DIR = f'{BASE_DIR}/data/'

    def __init__(self, raw_env_filename, processed_data_filename, top_species_percent):
        self.sequester_percent = 0.2
        self.test_percent = 0.2
        self.calibration_percent = 0.1
        self.processed_data_filename = processed_data_filename
        self.raw_env_filename = raw_env_filename
        self.top_species_percent = top_species_percent


    def generate_training_data(self, interaction_matrix, species_counts, top_species_idx, full_env_data):
            storage = dict()
            num_rows, num_cols = interaction_matrix.shape
            training_data = []
            env_data = []
            ground_truth_calibration_rows = []
            ground_truth_calibration_cols = []
            ground_truth_calibration_vals = []
            ground_truth_test_rows = []
            ground_truth_test_cols = []
            ground_truth_test_vals = []
            ground_truth_species = []
            sequestered_data = []
            complete_training = []
            complete_training_env = []
            training_data_row_count = 0
            for r in range(num_rows):
                row = interaction_matrix[r].tolist()[0]
                sequester_percent = random.random()
                if sequester_percent <= self.sequester_percent:
                    sequestered_data.append(row)
                    continue
                if full_env_data.any():
                    env_data.append(full_env_data[r])
                if random.random() <= self.test_percent:
                    is_calibration = False
                    if random.random() <= self.calibration_percent:
                        is_calibration = True
                    for c in range(len(row)):
                        if not c in top_species_idx:
                            if is_calibration:
                                ground_truth_calibration_rows.append(training_data_row_count)
                                ground_truth_calibration_cols.append(c)
                                ground_truth_calibration_vals.append(row[c])
                            else:
                                ground_truth_test_rows.append(training_data_row_count)
                                ground_truth_test_cols.append(c)
                                ground_truth_test_vals.append(row[c])
                            row[c] = 0
                    training_data.append(row)
                else:
                    complete_training.append(row)
                    if full_env_data.any():
                        complete_training_env.append(full_env_data[r])
                    training_data.append(row)
                training_data_row_count += 1
            storage['training_data'] = np.array(training_data)
            storage['ground_truth_calibration'] = np.array((ground_truth_calibration_rows, ground_truth_calibration_cols, ground_truth_calibration_vals), dtype=np.int32)
            storage['ground_truth_test'] = np.array((ground_truth_test_rows, ground_truth_test_cols, ground_truth_test_vals), dtype=np.int32)
            storage['sequestered_data'] = np.array(sequestered_data)
            storage['species_counts'] = species_counts
            storage['top_species_idx'] = top_species_idx
            storage['env_data'] = np.array(env_data)
            storage['complete_training'] = np.array(complete_training)
            storage['complete_training_env'] = np.array(complete_training_env)
            self._save_processed_data(storage)
            return storage

    def save_processed_data(self, data):
        filename = f'{self.PROCESSED_DATA_DIR}{self.processed_data_filename}'
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
            
    def load_raw_gbif_data(self, raw_data_filename):
        data_df = pd.read_csv(f'{self.RAW_DATA_DIR}{raw_data_filename}', usecols = ['decimalLatitude', 'decimalLongitude', 'speciesKey'])
        data_df["site"] = data_df["decimalLatitude"].map(str) + ' ' + data_df["decimalLongitude"].map(str)
        data_df['count'] = 1
        df_counts = data_df.groupby(['speciesKey']).count()
        df_counts["speciesKey"] = df_counts.index
        df_counts = df_counts.drop(columns=['decimalLatitude', 'decimalLongitude', 'site'])
        df_counts_list = df_counts.values.tolist()
        [df_counts_list, top_species_idx] = self._get_top_species(df_counts_list)
        df_to_pivot = data_df.pivot_table('count', index='site', columns='speciesKey').fillna(-1).astype(int)
        df_to_pivot = df_to_pivot.sample(frac=1)
        env_data = self._load_raw_env(df_to_pivot)
        interaction_matrix = csr_matrix(df_to_pivot.astype(pd.SparseDtype("int")).sparse.to_coo()).todense()
        return interaction_matrix, df_counts_list, top_species_idx, env_data

    def get_top_species(self, df_counts_list):
        for i in range(len(df_counts_list)):
            df_counts_list[i].append(i)
        df_counts_list.sort()
        top_species_list_threshold = round(self.top_species_percent * len(df_counts_list))
        top_species_idx = set()
        for item in df_counts_list[top_species_list_threshold:]:
            top_species_idx.add(item[2])
        return [df_counts_list, top_species_idx]
    
    def load_raw_env(self, data_df):
        if self.raw_env_filename == None:
            return []
        env_df = pd.read_csv(f'{self.RAW_DATA_DIR}{self.raw_env_filename}').fillna(0)
        env_df["site"] = env_df["decimalLatitude"].map(str) + ' ' + env_df["decimalLongitude"].map(str)
        env_df = env_df.merge(data_df, how='inner', on=['site'])
        env_df = env_df.iloc[:, 2:31] 
        numpy_version = env_df.to_numpy()
       
        return numpy_version