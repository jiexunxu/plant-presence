import matplotlib.pyplot as plt
import pandas as pd

from lib.train_evaluate_pipeline import run_pipeline, evaluate_maxent

train_params = {
    "n_estimators": 150,
    "max_depth": 6,
    "scale_pos_weight": 18,
    "learning_rate": 0.1,
    "subsample": 0.9,
}

presence_data_offset = 61
ignored_species=[0, 1, 2, 3, 4]
considered_species=list(range(3990, 4000, 1))
train_end = 3500
test_end = 4250

all_histogram_scores = {}
specific_histogram_scores = {}

print('======GBIF_maxent_sequestered======')
histogram_scores = evaluate_maxent(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)),
    presence_data_offset=presence_data_offset,
    maxent_filename='gbif_maxent_predictions.csv',
    data_file='gbif_combined_data.csv',
    ignored_species=ignored_species,
    considered_species=considered_species)
all_histogram_scores['Pure MaxEnt'] = histogram_scores
specific_histogram_scores['Pure MaxEnt'] = histogram_scores

print('======GBIF_xgb_baseline_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_xgb_baseline_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species)
all_histogram_scores['Pure XgBoost'] = histogram_scores

print('======GBIF_maxent_cascade_top1_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + [46],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_maxent_cascade_top1_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species)
all_histogram_scores['MaxEnt/XgBoost Top 1'] = histogram_scores

print('======GBIF_maxent_cascade_top5_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + list(range(46, 51)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_maxent_cascade_top5_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species)
all_histogram_scores['MaxEnt/XgBoost Top 5'] = histogram_scores
specific_histogram_scores['MaxEnt/XgBoost Top 5'] = histogram_scores

print('======GBIF_xgb_cascade_top1_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + [51],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_xgb_cascade_top1_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species)
all_histogram_scores['XgBoost/XgBoost Top 1'] = histogram_scores

print('======GBIF_xgb_cascade_top5_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + list(range(51, 56)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_xgb_cascade_top5_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species)
all_histogram_scores['XgBoost/XgBoost Top 5'] = histogram_scores

print('======GBIF_gold_standard_cascade_top1_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + [56],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_gold_standard_cascade_top1_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species)
all_histogram_scores['GroundTruth/XgBoost Top 1'] = histogram_scores

print('======GBIF_gold_standard_cascade_top5_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + list(range(56, 61)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_gold_standard_cascade_top5_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species)
all_histogram_scores['Ground Truth/XgBoost Top 5'] = histogram_scores
specific_histogram_scores['Ground Truth/XgBoost Top 5'] = histogram_scores

plt.rcParams.update({'font.size': 21})
print('======Histogram Jaccard scores for all methods======')
for key, value in all_histogram_scores.items():
    print(f"{key}: {value}")

df = pd.DataFrame(all_histogram_scores)
df.plot.bar(rot=0, figsize=(12, 8))
plt.tight_layout()
plt.legend(ncols=2, fontsize=17)
plt.show()

print('======Histogram Jaccard scores for selected methods======')
for key, value in specific_histogram_scores.items():
     print(f"{key}: {value}")

df = pd.DataFrame(specific_histogram_scores)
df.plot.bar(rot=0, figsize=(12, 5))
plt.tight_layout()
plt.legend(fontsize=17)
plt.show()