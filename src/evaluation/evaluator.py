import pandas as pd
import pathlib
import numpy as np
import random


class Evaluator:
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
    RAW_DATA_DIR = f'{BASE_DIR}/data/'
    RESULTS_DATA_DIR = f'{BASE_DIR}/results/'

    def __init__(self, input_data_file='gbif_env_presence_data.csv', presence_data_offset=33, sep=','):
        self.ground_truths = pd.read_csv(f'{self.RAW_DATA_DIR}{input_data_file}', sep=sep).iloc[:, presence_data_offset:].astype('int8')

    def load_result_csv(self, file_name='probabilities_xgboost_all_regions.csv', threshold=0.5, data_start=2500, data_end=3500, ignored_species=[], considered_species=None):
        predictions = pd.read_csv(f'{self.RESULTS_DATA_DIR}{file_name}', header=None)
        predictions = predictions[data_start:data_end]
        if considered_species is not None:
            predictions = predictions.iloc[:, considered_species]
        else:
            predictions = predictions.drop(predictions.columns[ignored_species], axis=1)
        
        raw_scores = predictions.copy(deep=True)
        predictions[predictions > threshold] = 10
        predictions[predictions <= threshold] = -10
        predictions = predictions.astype('int8')
        labels = self.ground_truths[data_start:data_end]
        if considered_species is not None:
            labels = labels.iloc[:, considered_species]
        else:
            labels = labels.drop(labels.columns[ignored_species], axis=1)

        return [predictions, labels, raw_scores]

    def calculate_scores(self, predictions, ground_truths):
        if predictions.shape != ground_truths.shape:
            raise Exception(f'Predictions dimension {predictions.shape} and ground truths dimension {ground_truths.shape} does not match!')
        [unique, counts] = np.unique(predictions.to_numpy().flatten() + ground_truths.to_numpy().flatten(), return_counts=True)
        prediction_results = dict(zip(unique, counts))
        true_negative = prediction_results.get(-11, 0)
        false_negative = prediction_results.get(-9, 0)
        false_positive  = prediction_results.get(9, 0)
        true_positive = prediction_results.get(11, 0)

        accuracy = self._round_division(true_positive + true_negative, true_positive + true_negative + false_positive + false_negative  )
        precision = self._round_division(true_positive, true_positive + false_positive)
        recall = self._round_division(true_positive, true_positive + false_negative)
        jaccard_similarity = self._round_division(true_positive, true_positive + false_positive + false_negative)
        specificity = self._round_division(true_negative, true_negative + false_positive)
        f1_score = self._round_division(2 * precision * recall, precision + recall)
        print(f'True Positives: {true_positive}, False Positives: {false_positive}, True Negatives: {true_negative}, False Negatives: {false_negative}, Total: {true_positive + false_positive + true_negative + false_negative}')
        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Specificity: {specificity}, F1 (Dice) Score: {f1_score}, Jaccard Similarity: {jaccard_similarity}')

    def calculate_histogram_scores(self, predictions, ground_truths, histogram_size):
        f1_scores = []
        species_count = len(predictions.columns)
        for i in range(0, species_count, histogram_size):
            histogram_idx = list(range(i, min(i + histogram_size, species_count)))
            predictions_histogram = predictions.iloc[:, histogram_idx]
            ground_truths_histogram = ground_truths.iloc[:, histogram_idx]
            [unique, counts] = np.unique(predictions_histogram.to_numpy().flatten() + ground_truths_histogram.to_numpy().flatten(), return_counts=True)
            prediction_results = dict(zip(unique, counts))
            true_negative = prediction_results.get(-11, 0)
            false_negative = prediction_results.get(-9, 0)
            false_positive  = prediction_results.get(9, 0)
            true_positive = prediction_results.get(11, 0)
            precision = self._round_division(true_positive, true_positive + false_positive)
            recall = self._round_division(true_positive, true_positive + false_negative)
            f1_score = self._round_division(2 * precision * recall, precision + recall)
            f1_scores.append(f1_score)
        
        return f1_scores


    def p_value_test(self, n1, n2, num_iterations = 10000):
        ndiff = sum([ n1[i]-n2[i] for i in range(len(n1)) ])
        if ndiff < 0:
            nbig = n2
            nsmall = n1
            ndiff = -ndiff
        else:
            nbig = n1
            nsmall = n2
        
        bcount = 0
        for niter in range(num_iterations):
            tdiff = 0
            for i in range(len(nbig)):
                a = random.random()
                if a <0.5:
                    tdiff += nsmall[i]-nbig[i]
                else:
                    tdiff += nbig[i]-nsmall[i]
            if tdiff > ndiff:
                bcount += 1
        p_value = float(bcount) / num_iterations
        print(f'p value: {p_value}')
        return p_value


    def compare_predictions(self, predictions1, predictions2, ground_truths, model1_name = 'model1', model2_name = 'model2'):
        if predictions1.shape != ground_truths.shape or predictions2.shape != ground_truths.shape:
            raise Exception(f'Predictions dimension {predictions1.shape} and {predictions2.shape}, and ground truths dimension {ground_truths.shape} does not match!')
        presence_both_correct = 0
        presence_pred1_correct = 0
        presence_pred2_correct = 0
        presence_both_incorrect = 0
        absence_both_correct = 0
        absence_pred1_correct = 0
        absence_pred2_correct = 0
        absence_both_incorrect = 0
        presence_count = 0
        absence_count = 0
        for entry in zip(predictions1.to_numpy().flatten(), predictions2.to_numpy().flatten(), ground_truths.to_numpy().flatten()):
            prediction1 = entry[0]
            prediction2 = entry[1]
            truth = entry[2]
            if truth == 1:
                presence_count += 1
                if prediction1 == 10 and prediction2 == 10:
                    presence_both_correct += 1
                elif prediction1 == 10 and prediction2 == -10:
                    presence_pred1_correct += 1
                elif prediction1 == -10 and prediction2 == 10:
                    presence_pred2_correct += 1
                else:
                    presence_both_incorrect += 1
            else:
                absence_count += 1
                if prediction1 == -10 and prediction2 == -10:
                    absence_both_correct += 1
                elif prediction1 == -10 and prediction2 == 10:
                    absence_pred1_correct += 1
                elif prediction1 == 10 and prediction2 == -10:
                    absence_pred2_correct += 1
                else:
                    absence_both_incorrect += 1
        print(f'Comparing models {model1_name} and {model2_name}:')
        print(f'When a species is present in ground truth test data({presence_count} occurences), both models correct: {presence_both_correct}({self._get_percentage(presence_both_correct, presence_count)}), only {model1_name} correct: {presence_pred1_correct}({self._get_percentage(presence_pred1_correct, presence_count)}), only {model2_name} correct: {presence_pred2_correct}({self._get_percentage(presence_pred2_correct, presence_count)}), both incorrect: {presence_both_incorrect}({self._get_percentage(presence_both_incorrect, presence_count)})')
        print(f'When a species is absent in ground truth test data({absence_count} occurences), both models correct: {absence_both_correct}({self._get_percentage(absence_both_correct, absence_count)}), only {model1_name} correct: {absence_pred1_correct}({self._get_percentage(absence_pred1_correct, absence_count)}), only {model2_name} correct: {absence_pred2_correct}({self._get_percentage(absence_pred2_correct, absence_count)}), both incorrect: {absence_both_incorrect}({self._get_percentage(absence_both_incorrect, absence_count)})')


    def _round_division(self, dividend, divisor, precision=5):
        if divisor == 0:
            return 0
        return round(dividend / divisor, precision)

    def _get_percentage(self, dividend, divisor):
        percentage = 100 * dividend / divisor
        return '{:0.2f}%'.format(percentage)