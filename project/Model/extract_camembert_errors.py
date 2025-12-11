"""
Script to extract CamemBERT model's incorrect predictions
and organize them by group
"""

import pandas as pd
import os

# Load original dataset to get line numbers
original_dataset_file = 'data/crows_pairs_fr_final.csv'
original_df = pd.read_csv(original_dataset_file)
original_df['original_line_number'] = original_df.index

# Load CamemBERT model results
results_file = 'Model/result_output_camembert/crows_fr_trained/crows_fr/full_results.csv'
results_df = pd.read_csv(results_file)

# Merge with original dataset to get original line numbers
# Use text as join key
results_df = results_df.merge(original_df[['text', 'original_line_number']], on='text', how='left')

print(f"Total examples: {len(results_df)}")
print(f"Actual label distribution:\n{results_df['actual_label'].value_counts()}")

# Filter incorrect predictions
errors_df = results_df[results_df['predicted_label'] != results_df['actual_label']].copy()

print(f"\nNumber of errors: {len(errors_df)}")
print(f"Error rate: {len(errors_df) / len(results_df) * 100:.2f}%")

# Reorganize columns to put original_line_number first
cols = ['original_line_number'] + [col for col in errors_df.columns if col != 'original_line_number']
errors_df = errors_df[cols]

# Sort by original line number
errors_df = errors_df.sort_values('original_line_number')

# Display error statistics by group
print("\n" + "="*60)
print("Error statistics by group:")
print("="*60)
group_stats = errors_df.groupby('group').size().reset_index(name='error_count')
group_stats = group_stats.sort_values('error_count', ascending=False)
print(group_stats.to_string(index=False))

# Save to new CSV file
output_dir = 'Model/result_output_camembert/crows_fr_trained/crows_fr'
output_file = os.path.join(output_dir, 'errors_by_group.csv')
errors_df.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print(f"Errors saved to: {output_file}")
print(f"{'='*60}")

# Also create a summary of errors by group
summary_file = os.path.join(output_dir, 'errors_summary_by_group.csv')
group_summary = errors_df.groupby('group').agg({
    'text': 'count',
    'predicted_probability': 'mean'
}).rename(columns={
    'text': 'error_count',
    'predicted_probability': 'average_probability'
}).reset_index()

group_summary = group_summary.sort_values('error_count', ascending=False)
group_summary.to_csv(summary_file, index=False)

print(f"Summary saved to: {summary_file}")
print(f"\nSummary overview by group:")
print(group_summary.to_string(index=False))
