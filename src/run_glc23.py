import matplotlib.pyplot as plt
import pandas as pd

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
considered_species=list(range(1164, 1174, 1))
train_end = 4012
test_end = 4525

all_histogram_scores = {}
specific_histogram_scores = {}

print('======GLC23_maxent_sequestered======')
histogram_scores = evaluate_maxent(
    train_end=train_end,
    test_end=test_end, 
    feature_columns=list(range(2, 55)),
    presence_data_offset=presence_data_offset,
    maxent_filename='glc23_maxent_predictions.csv',
    data_file='glc23_combined_data.csv',
    ignored_species=ignored_species, 
    considered_species=considered_species,
    histogram_width=100)
all_histogram_scores['Pure MaxEnt'] = histogram_scores
specific_histogram_scores['Pure MaxEnt'] = histogram_scores

print('======GLC23_xgb_baseline_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_xgb_baseline_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['Pure XgBoost'] = histogram_scores

print('======GLC23_maxent_cascade_top5_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(60, 65)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_maxent_cascade_top5_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['MaxEnt/XgBoost Top 5'] = histogram_scores
specific_histogram_scores['MaxEnt/XgBoost Top 5'] = histogram_scores

print('======GLC23_po_maxent_cascade_top5_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(55, 60)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_maxent_cascade_top5_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['MaxEnt+PO/XgBoost Top 5'] = histogram_scores

print('======GLC23_po_maxent_cascade_top5_sequestered_plus_po======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(55, 60)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_maxent_cascade_top5_sequestered_with_po.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['MaxEnt+PO/XgBoost+PO Top 5'] = histogram_scores
specific_histogram_scores['MaxEnt+PO/XgBoost+PO Top 5'] = histogram_scores

print('======GLC23_xgb_cascade_top5_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(65, 70)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_xgb_cascade_top5_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['XgBoost/XgBoost Top 5'] = histogram_scores

print('======GLC23_gold_standard_cascade_top5_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(70, 75)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_gold_standard_cascade_top5_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['Ground Truth/XgBoost Top 5'] = histogram_scores

print('======GLC23_gold_standard_cascade_top5_sequestered_plus_po======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(70, 75)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_gold_standard_cascade_top5_sequestered_with_po.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['Ground Truth/XgBoost+PO Top 5'] = histogram_scores
specific_histogram_scores['Ground Truth/XgBoost+PO Top 5'] = histogram_scores

print('======GLC23_po_cascade_top5_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(75, 80)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_cascade_top5_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['PO/XgBoost Top 5'] = histogram_scores

print('======GLC23_po_xgb_cascade_top5_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + list(range(80, 85)),
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_xgb_cascade_top5_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['XgBoost+PO/XgBoost Top 5'] = histogram_scores

print('======GLC23_po_maxent_cascade_top1_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [55],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_maxent_cascade_top1_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['MaxEnt+PO/XgBoost Top 1'] = histogram_scores

print('======GLC23_po_xgb_cascade_top1_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [80],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_xgb_cascade_top1_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['XgBoost+PO/XgBoost Top 1'] = histogram_scores

print('======GLC23_po_cascade_top1_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [75],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_po_cascade_top1_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['PO/XgBoost Top 1'] = histogram_scores

print('======GLC23_gold_standard_cascade_top1_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [70],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_gold_standard_cascade_top1_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['Ground Truth/XgBoost Top 1'] = histogram_scores

print('======GLC23_maxent_cascade_top1_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [60],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_maxent_cascade_top1_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['MaxEnt/XgBoost Top 1'] = histogram_scores

print('======GLC23_xgb_cascade_top1_sequestered======')
histogram_scores = run_pipeline(
    train_end=train_end, 
    test_end=test_end, 
    feature_columns=list(range(2, 55)) + [65],
    presence_data_offset=presence_data_offset,
    train_params=train_params,
    data_file='glc23_combined_data.csv',
    output_file='GLC23_xgb_cascade_top1_sequestered.csv',
    ignored_species=ignored_species,
    considered_species=considered_species,
    histogram_width=100,
    use_po_data=True,)
all_histogram_scores['XgBoost/XgBoost Top 1'] = histogram_scores

plt.rcParams.update({'font.size': 21})
print('======Histogram Jaccard scores for all methods======')
for key, value in all_histogram_scores.items():
     print(f"{key}: {value}")

df = pd.DataFrame(all_histogram_scores)
df.plot.bar(rot=0, figsize=(12, 8))
plt.tight_layout()
plt.legend(ncols=2, fontsize=14)
plt.show()

print('======Histogram Jaccard scores for selected methods======')
for key, value in specific_histogram_scores.items():
     print(f"{key}: {value}")

df = pd.DataFrame(specific_histogram_scores)
df.plot.bar(rot=0, figsize=(12, 5))
plt.tight_layout()
plt.legend(fontsize=17)
plt.show()