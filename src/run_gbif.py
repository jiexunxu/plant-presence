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
train_end = 3500
test_end = 4250

print('======GBIF_maxent_sequestered======')

evaluate_maxent(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)),
    presence_data_offset=presence_data_offset,
    maxent_filename='gbif_maxent_predictions.csv',
    data_file='gbif_combined_data.csv',
    ignored_species=ignored_species
)

print('======GBIF_xgb_baseline_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_xgb_baseline_sequestered.csv',
    ignored_species=ignored_species)

print('======GBIF_maxent_cascade_top1_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + [46],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_maxent_cascade_top1_sequestered.csv',
    ignored_species=ignored_species)

print('======GBIF_maxent_cascade_top5_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + list(range(46, 51)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_maxent_cascade_top5_sequestered.csv',
    ignored_species=ignored_species)

print('======GBIF_xgb_cascade_top1_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + [51],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_xgb_cascade_top1_sequestered.csv',
    ignored_species=ignored_species)

print('======GBIF_xgb_cascade_top5_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + list(range(51, 56)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_xgb_cascade_top5_sequestered.csv',
    ignored_species=ignored_species)

print('======GBIF_gold_standard_cascade_top1_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + [56],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_gold_standard_cascade_top1_sequestered.csv',
    ignored_species=ignored_species)

print('======GBIF_gold_standard_cascade_top5_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 46)) + list(range(56, 61)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='gbif_combined_data.csv',
    output_file='GBIF_gold_standard_cascade_top5_sequestered.csv',
    ignored_species=ignored_species)
