from data.data_loader import DataLoader
from evaluation.evaluator import Evaluator
from model.xgboost_model import XGBoostModel

def run_pipeline(train_end, test_end, feature_columns, presence_data_offset, train_params, data_file, output_file, ignored_species):
    loader = DataLoader(input_data_file=data_file, feature_columns=feature_columns, presence_data_offset=presence_data_offset, sep=';')

    [train_data, train_labels] = loader.extract_training_data(data_start=0, data_end=train_end)
    test_data = loader.extract_test_data(data_start=train_end, data_end=test_end)

    model = XGBoostModel() 
    model.train_evaluate_and_output(train_data, train_labels, test_data, output_file, params=train_params)

    print('Evaluating all species:')
    evaluator = Evaluator(input_data_file=data_file, presence_data_offset=presence_data_offset, sep=';')
    [predictions, ground_truths, _] = evaluator.load_result_csv(file_name=output_file, threshold=0.5, data_start=train_end, data_end=test_end, ignored_species=[])
    evaluator.calculate_scores(predictions, ground_truths)

    print('Evaluating all but the top 5 species:')
    [predictions, ground_truths, _] = evaluator.load_result_csv(file_name=output_file, threshold=0.5, data_start=train_end, data_end=test_end, ignored_species=ignored_species)
    evaluator.calculate_scores(predictions, ground_truths)

def evaluate_maxent(train_end, test_end, feature_columns, presence_data_offset, maxent_filename, data_file, ignored_species):
    loader = DataLoader(input_data_file=data_file, feature_columns=feature_columns, presence_data_offset=presence_data_offset, sep=';')
    evaluator = Evaluator(input_data_file=data_file, presence_data_offset=presence_data_offset, sep=';')

    print('Evaluating all species:')
    [ground_truths, maxent_predictions] = loader.load_maxent_predictions(data_start=train_end, data_end=test_end, maxent_filename=maxent_filename)
    evaluator.calculate_scores(maxent_predictions, ground_truths)

    print('Evaluating all but the top 5 species:')
    [ground_truths, maxent_predictions] = loader.load_maxent_predictions(data_start=train_end, data_end=test_end, maxent_filename=maxent_filename, ignored_species=ignored_species)
    evaluator.calculate_scores(maxent_predictions, ground_truths)