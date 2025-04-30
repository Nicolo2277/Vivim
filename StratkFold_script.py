import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_stratified_group_folds(metadata_path,
                                 data_root,
                                 n_splits=5,
                                 random_state=42,
                                 frame_bin_strategy='quantile',
                                 n_bins=5):

    df_meta = pd.read_csv(metadata_path)

    frame_counts = []
    for item in df_meta['item'].unique():
        item_path = os.path.join(data_root, str(item))
        if os.path.isdir(item_path):
            # assume frames are image files inside directory
            n_frames = len([f for f in os.listdir(item_path)
                            if os.path.isfile(os.path.join(item_path, f))])
        elif os.path.isfile(item_path):
            # single file video or image counts as 1 frame
            n_frames = 1
        else:
            # missing item
            n_frames = 0
        frame_counts.append({'item': item, 'n_frames': n_frames})
    df_frames = pd.DataFrame(frame_counts)

    # Merge frame counts into metadata and filter out missing items
    df = df_meta.merge(df_frames, on='item', how='inner')

    # Aggregate per clinical_case
    agg = df.groupby('clinical_case').agg(
        histological=pd.NamedAgg(column='histological', aggfunc=lambda s: s.mode().iat[0]),
        total_frames=pd.NamedAgg(column='n_frames', aggfunc='sum')
    ).reset_index()

    # Discretize frame counts into bins for stratification
    if frame_bin_strategy == 'quantile':
        agg['frame_bin'] = pd.qcut(agg['total_frames'], q=n_bins, labels=False, duplicates='drop')
    else:  # uniform
        agg['frame_bin'] = pd.cut(agg['total_frames'], bins=n_bins, labels=False)

    # Create combined stratification label
    agg['strata'] = agg['histological'].astype(str) + '_bin' + agg['frame_bin'].astype(str)

    # Prepare stratifier
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, val_idx in skf.split(agg['clinical_case'], agg['strata']):
        train_cases = agg.loc[train_idx, 'clinical_case'].tolist()
        val_cases = agg.loc[val_idx, 'clinical_case'].tolist()
        folds.append((train_cases, val_cases))

    return df, folds

# Example usage:
# df, folds = make_stratified_group_folds(
#     metadata_path='histology.csv',
#     data_root='/path/to/frames_or_videos',
#     n_splits=5,
#     random_state=0,
#     frame_bin_strategy='quantile',
#     n_bins=5
# )
# for i, (tr, val) in enumerate(folds):
#     print(f"Fold {i}: train cases={len(tr)}, val cases={len(val)}")
