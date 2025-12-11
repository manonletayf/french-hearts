"""
Script to transform the CrowS-Pairs FR dataset for the ALBERT/CamemBERT baseline

TRANSFORMATIONS:
1. Sentence selection:
   - If stereo_antistereo == 'stereo' → keep sent_more
   - If stereo_antistereo == 'antistereo' → keep sent_less

2. Label: all sentences → 'stereotype'

3. Group: preserve original bias_type categories (no mapping)
"""

import pandas as pd

# Load the dataset
df = pd.read_csv('crows_pairs_FR_languagearc_contribution+210.csv', sep='\t')

# Select the sentence containing the stereotype
df['text'] = df.apply(
    lambda row: row['sent_more'] if row['stereo_antistereo'] == 'stereo' else row['sent_less'],
    axis=1
)

# Create the label
df['label'] = 'stereotype'

# Preserve original bias categories
df['group'] = df['bias_type']

# Create the final dataframe
df_final = pd.DataFrame({
    'text': df['text'],
    'label': df['label'],
    'group': df['group']
}).dropna()

# Save
output_file = 'crows_pairs_fr_converted.csv'
df_final.to_csv(output_file, index=False)

print(f"Conversion completed: {len(df_final)} sentences")
print(f"File created: {output_file}")
print(f"\nGroup distribution:")
print(df_final['group'].value_counts())
