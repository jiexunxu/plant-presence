import pandas as pd
import pathlib


class DataLoader:
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
    RAW_DATA_DIR = f'{BASE_DIR}/data/'
    RESULTS_DATA_DIR = f'{BASE_DIR}/results/'

    def __init__(self, input_data_file='gbif_env_presence_data.csv', feature_columns=[], presence_data_offset=33, sep=','):
        self.gbif_env_presence_data = pd.read_csv(f'{self.RAW_DATA_DIR}{input_data_file}', sep=sep)
        self.feature_columns = feature_columns
        self.presence_data_offset = presence_data_offset

    def load_presence_only_data(self, input_po_file='glc23_presence_only_all_species.csv'):
        po_df = pd.read_csv(f'{self.RAW_DATA_DIR}{input_po_file}', sep=';')
        po_df = po_df.iloc[:, 1:]
        return po_df

    def load_maxent_predictions(self, data_start, data_end, maxent_filename, ignored_species=None, considered_species=None):
        maxent_predictions = pd.read_csv(f'{self.RAW_DATA_DIR}{maxent_filename}', sep=';')
        ground_truth = self.gbif_env_presence_data.iloc[data_start:data_end, self.presence_data_offset:]
        maxent_predictions[maxent_predictions > 0.5] = 10
        maxent_predictions[maxent_predictions <= 0.5] = -10
        maxent_predictions = maxent_predictions.astype('int8')
        if considered_species is not None:
            maxent_predictions = maxent_predictions.iloc[:, considered_species]
            ground_truth = ground_truth.iloc[:, considered_species]
        elif ignored_species != None:
            maxent_predictions.drop(maxent_predictions.columns[ignored_species],axis=1,inplace=True)
            ground_truth.drop(ground_truth.columns[ignored_species],axis=1,inplace=True)

        return [ground_truth, maxent_predictions]
    
    def extract_training_data(self, data_start=0, data_end=1500, species_idx_as_feature=[], maxent_filename=None):
        df = self.gbif_env_presence_data.iloc[data_start:data_end]
        use_maxent = False
        maxent_data = []
        if maxent_filename != None:
            maxent_data = self._read_maxent_data(maxent_filename)
            use_maxent = True
        return self._split_feature_and_label(df, 'Base Model Training Data', species_idx_as_feature, use_maxent, maxent_data)

    def extract_test_data(self, data_start=2500, data_end=3500, species_idx_as_feature=[], maxent_filename=None):
        df = self.gbif_env_presence_data.iloc[data_start:data_end]
        use_maxent = False
        maxent_data = []
        if maxent_filename != None:
            maxent_data = self._read_maxent_data(maxent_filename)
            maxent_data.index = maxent_data.index + data_start
            use_maxent = True
        return self._split_feature_and_label(df, 'Test Data', species_idx_as_feature, use_maxent, maxent_data)[0]

    def get_data_stats(self, top_species_list_threshold=100):
        all_labels = self.gbif_env_presence_data.iloc[:, self.presence_data_offset:].astype('int8')
        all_labels[all_labels < 0] = 0
        species_presence_count = all_labels.sum(axis=0).reset_index(drop=True).sort_values(ascending=False)
        top_species_idx = species_presence_count.head(top_species_list_threshold).index.to_list()
        top_species_ids = all_labels.columns[top_species_idx].to_list
        return { 'top_species_idx': top_species_idx, 'top_species_ids': top_species_ids }

    def _split_feature_and_label(self, df, description='', species_idx_as_feature=[], use_maxent=False, maxent_data=None):
        features = df.iloc[:, self.feature_columns]
        labels = df.iloc[:, self.presence_data_offset:]
        labels[labels < 0] = 0
        if use_maxent:
            features = features.iloc[: , :-len(species_idx_as_feature)]
            maxent_data = maxent_data.iloc[:, :len(species_idx_as_feature)]
            features = pd.concat([features, maxent_data], axis=1, ignore_index=True)
            features = features.fillna(0.0)
        print(f'Loaded features with dimension: {features.shape} and labels with dimension: {labels.shape} to be used as: {description}')
        return [features, labels]
