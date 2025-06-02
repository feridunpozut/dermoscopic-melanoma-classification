'''
Combines the ISIC 2018, 2019, and 2020 training datasets into a single dataset for binary classification.
'''
import pandas as pd
import os

def combine_isic_datasets(img_path) -> (pd.Series, pd.Series, pd.Series):
    """
    Combines the ISIC 2018, 2019, and 2020 training datasets into a single dataset for binary classification.
    
    The function performs the following steps:
      - Reads the groundtruth CSV files for ISIC 2018, 2019 and 2020.
      - Assigns the corresponding dataset names to each.
      - Renames columns if necessary (e.g., 'image_name' to 'image' for ISIC 2020).
      - Drops rows where the 'image' column contains '_downsampled'.
      - For melanoma ('melanoma' label):
          * Filters rows from ISIC 2019 where 'MEL' == 1.0.
          * Filters rows from ISIC 2020 where 'benign_malignant' == 'malignant'.
          * Filters rows from ISIC 2018 where 'MEL' == 1.0.
      - For benign ('benign' label):
          * Samples rows with 'NV'==1.0 from ISIC 2018 and ISIC 2019 and
            rows with 'benign_malignant'=='benign' from ISIC 2020 based on
            the number of melanoma images per dataset.
      - Drops duplicate entries.
      - Returns the combined dataset's 'image', 'label', and 'dataset' columns.
    
    Returns:
        combined_images (pd.Series): Series with image names.
        combined_labels (pd.Series): Series with corresponding labels ('melanoma' or 'benign').
        combined_datasets (pd.Series): Series with corresponding dataset names.
    """
    
    csv_18 = pd.read_csv('../data/splits/ISIC2018_Task3_Training_GroundTruth.csv')
    csv_18['dataset'] = 'ISIC_2018'
    csv_19 = pd.read_csv('../data/splits/ISIC_2019_Training_GroundTruth.csv')
    csv_19['dataset'] = 'ISIC_2019'
    csv_20 = pd.read_csv('../data/splits/ISIC_2020_Training_GroundTruth.csv')
    csv_20['dataset'] = 'ISIC_2020'
    csv_20.rename(columns={'image_name': 'image'}, inplace=True)

    csv_18.drop(index=csv_18[csv_18['image'].str.contains('_downsampled')].index, inplace=True)
    csv_19.drop(index=csv_19[csv_19['image'].str.contains('_downsampled')].index, inplace=True)
    csv_20.drop(index=csv_20[csv_20['image'].str.contains('_downsampled')].index, inplace=True)
    
    combined_mel = pd.concat([
        csv_19.loc[csv_19['MEL'] == 1.0, ['image','dataset']],
        csv_20.loc[csv_20['benign_malignant'] == 'malignant', ['image','dataset']],
        csv_18.loc[csv_18['MEL'] == 1.0, ['image','dataset']]
    ]).reset_index(drop=True)

    combined_mel = pd.DataFrame(combined_mel, columns=['image', 'dataset'])
    combined_mel.drop_duplicates(subset=['image'],inplace=True)
    combined_mel['label'] = 1.0
    
    combined_nv = pd.concat([
        csv_19[csv_19['NV']==1.0].sample(combined_mel['dataset'].value_counts()['ISIC_2019'], random_state=42)[['image','dataset']],
        csv_20[csv_20['benign_malignant']=='benign'].sample(combined_mel['dataset'].value_counts()['ISIC_2020'], random_state=42)[['image','dataset']]
    ]).reset_index(drop=True)
    
    combined_nv = pd.DataFrame(combined_nv, columns=['image', 'dataset'])
    combined_nv.drop_duplicates(subset=['image'],inplace=True)
    combined_nv['label'] = 0.0
    
    combined_dataset = pd.concat([combined_mel, combined_nv], ignore_index=True)
    combined_dataset.reset_index(drop=True, inplace=True)
    combined_dataset['image_path'] = combined_dataset['image'].apply(lambda x: os.path.join(img_path,x+'.jpg')
    train_val_dataset, test_dataset = train_test_split(combined_dataset, test_size=0.2, random_state=42, stratify=combined_dataset['label'])
    train_val_dataset.reset_index(drop=True, inplace=True)
    test_dataset.reset_index(drop=True, inplace=True)
    train_val_dataset.to_csv('../data/splits/train_val_dataset.csv', index=False)
    test_dataset.to_csv('../data/splits/test_dataset.csv', index=False)
    combined_dataset.to_csv('../data/splits/combined_dataset.csv', index=False)

    return combined_dataset['image'], combined_dataset['label'], combined_dataset['dataset']

if __name__ == "__main__":
    images, labels, datasets = combine_isic_datasets(img_path='../data/processed/artefact_removal')
    print(f"Combined dataset size: {len(images)}")
    print(f"Unique labels: {labels.unique()}")
    print(f"Unique datasets: {datasets.unique()}")