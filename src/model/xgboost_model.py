from os.path import exists
import xgboost as xgb
import pandas as pd
import pathlib


class XGBoostModel:
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
    RESULTS_DATA_DIR = f'{BASE_DIR}/results/'

    def train_evaluate_and_output(self, training_data, training_labels, test_data, output_csv, format='regular', params=None):
        if exists(f'{self.RESULTS_DATA_DIR}{output_csv}'):
            print(f'XGBoost: Result file {output_csv} already exists, skipping...')
            return
        
        if len(training_data.columns) != len(test_data.columns):
            raise Exception(f'XGBoost: Training data #column={len(training_data.columns)}, test data #columns={len(test_data.columns)}, they don\'t match!')
        test_data.set_axis(training_data.columns, axis=1, inplace=True)
        if params != None:
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBClassifier(n_estimators=150, max_depth=6, tree_method="gpu_hist", learning_rate=0.1, subsample=0.9, scale_pos_weight=21)
        
        model.fit(training_data, training_labels)
        full_data = pd.concat([training_data, test_data], axis=0, ignore_index=True).astype(float)
        presence_probabilities = model.predict_proba(full_data)
        if format == 'regular':
            pd.DataFrame(presence_probabilities).to_csv(f'{self.RESULTS_DATA_DIR}{output_csv}', index=False, header=False, float_format='%.5f')
        elif format == 'GLC2023':
            presence_probabilities = presence_probabilities[training_data.shape[0]:, :]
            presence_probabilities[presence_probabilities >= 0.5] = 1
            presence_probabilities[presence_probabilities < 0.5] = 0
            headers = training_labels.columns.to_list()
            headers_str = [str(x) for x in headers]
            df = pd.DataFrame(presence_probabilities, columns=headers_str)
            result = df.apply(lambda row:" ".join(row[row==1].index.to_list()), axis=1)
            result = result.reset_index().drop(columns=['index'])
            result.insert(loc=0, column="Id", value=result.reset_index().index+1)
            result.to_csv(f'{self.RESULTS_DATA_DIR}{output_csv}', index=False, header=['Id', 'Predicted'])
