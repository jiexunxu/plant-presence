from lib.train_evaluate_pipeline import run_pipeline, evaluate_maxent

train_params = {
    "n_estimators": 150,
    "max_depth": 6,
    "scale_pos_weight": 21,
    "learning_rate": 0.1,
    "subsample": 0.9,
}

presence_data_offset = 85
ignored_species=[0, 1, 2, 3, 4]
train_end = 4012
test_end = 4525

print('======GLC23_maxent_sequestered======')

evaluate_maxent(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)),
    presence_data_offset=presence_data_offset,
    maxent_filename='glc23_maxent_predictions.csv',
    data_file='glc23_combined_data.csv',
    ignored_species=ignored_species
)

print('======GLC23_xgb_baseline_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_xgb_baseline_sequestered.csv',
    ignored_species=ignored_species)


print('======GLC23_po_maxent_cascade_top1_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [55],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_maxent_cascade_top1_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_po_maxent_cascade_top5_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(55, 60)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_maxent_cascade_top5_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_maxent_cascade_top1_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [60],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_maxent_cascade_top1_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_maxent_cascade_top5_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(60, 65)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_maxent_cascade_top5_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_xgb_cascade_top1_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [65],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_xgb_cascade_top1_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_xgb_cascade_top5_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(65, 70)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_xgb_cascade_top5_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_gold_standard_cascade_top1_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [70],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_gold_standard_cascade_top1_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_gold_standard_cascade_top5_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(70, 75)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_gold_standard_cascade_top5_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_po_cascade_top1_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [75],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_cascade_top1_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_po_cascade_top5_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(75, 80)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_cascade_top5_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_po_xgb_cascade_top1_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [80],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_cascade_top1_sequestered.csv',
    ignored_species=ignored_species)

print('======GLC23_po_xgb_cascade_top5_sequestered======')

run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(80, 85)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_cascade_top5_sequestered.csv',
    ignored_species=ignored_species)
