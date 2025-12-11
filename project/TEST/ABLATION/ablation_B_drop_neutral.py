# ABLATION B: drop_neutral=True
# Removes all 'neutral' sentences from the dataset

from baseline_camembert_ablation import data_loader, train_model, evaluate_model

if __name__ == "__main__":
    print("="*80)
    print("ABLATION B: drop_neutral=True")
    print("="*80)

    train_data, test_data = data_loader(
        csv_file_path='../../data/crows_pairs_fr_final.csv',
        labelling_criteria='stereotype',
        dataset_name='CrowS_FR',
        sample_size=1000000,
        num_examples=5,
        drop_neutral=True
    )

    train_model(
        train_data,
        model_path='camembert-base',
        batch_size=32,
        epoch=4,
        learning_rate=2e-5,
        model_output_base_dir='../../model_output_ablation_drop_neutral',
        dataset_name='crows_fr_drop_neutral',
        seed=42,
        max_length=512
    )

    evaluate_model(
        test_data,
        model_output_dir='../../model_output_ablation_drop_neutral/crows_fr_drop_neutral',
        result_output_base_dir='../../result_output_ablation_drop_neutral',
        dataset_name='crows_fr',
        seed=42,
        max_length=512
    )
