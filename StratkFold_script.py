import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def gather_annotated_frames(input_root: Path):
    """
    Find all pairs of (frame.png, background.png) under input_root,
    grouped by clinical_case (top-level folder).

    Returns:
        DataFrame with columns [clinical_case, frame_path, gt_path, item]
            where `item` is the relative subfolder path under clinical_case.
    """
    records = []
    for dirpath, _, filenames in os.walk(input_root):
        files = set(f.lower() for f in filenames)
        if 'frame.png' in files and 'background.png' in files:
            dirp = Path(dirpath)
            rel = dirp.relative_to(input_root)
            clinical_case = rel.parts[0]
            item = rel.as_posix()
            records.append({
                'clinical_case': clinical_case,
                'item': item,
                'frame_path': dirp / 'frame.png',
                'gt_path': dirp / 'background.png'
            })
    return pd.DataFrame(records)


from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def make_stratified_folds_and_copy(
    input_root: Path,
    output_root: Path,
    n_splits: int = 5,
    random_state: int = 42,
    frame_bins: int = 4
):
    df = gather_annotated_frames(input_root)

    # build case_df with histology + frame_count + count_bin + strat_label
    cases = []
    for case in df['clinical_case'].unique():
        meta = input_root / case / 'metadata.csv'
        hist = pd.read_csv(meta)['histological'].mode().iat[0] if meta.exists() else 'unknown'
        cases.append({'clinical_case': case, 'histological': hist})
    case_df = pd.DataFrame(cases)

    # add frame counts and bins
    frame_counts = df.groupby('clinical_case').size().rename('frame_count').reset_index()
    case_df = case_df.merge(frame_counts, on='clinical_case')
    case_df['count_bin'] = pd.qcut(
        case_df['frame_count'], q=frame_bins,
        labels=False, duplicates='drop'
    )
    case_df['strat_label'] = (
        case_df['histological'].astype(str)
        + '_bin' + case_df['count_bin'].astype(str)
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(case_df['clinical_case'], case_df['strat_label']))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_cases = case_df.loc[train_idx, 'clinical_case']
        val_cases   = case_df.loc[val_idx,   'clinical_case']

        # ** count frames per split **
        train_sub = df[df['clinical_case'].isin(train_cases)]
        val_sub   = df[df['clinical_case'].isin(val_cases)]
        print(f"Fold {fold_idx}: "
              f"train cases={len(train_cases)}, "
              f"train frames={len(train_sub)}; "
              f"val cases={len(val_cases)}, "
              f"val frames={len(val_sub)}")

        # now copy
        fold_dir = output_root / f'fold_{fold_idx}'
        for split, cases_list in [('train', train_cases), ('val', val_cases)]:
            tgt = fold_dir / split
            tgt.mkdir(parents=True, exist_ok=True)
            subset = df[df['clinical_case'].isin(cases_list)]
            for _, row in subset.iterrows():
                dest = tgt / row['clinical_case'] / Path(row['item'])
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(row['frame_path'], dest / 'frame.png')
                shutil.copy2(row['gt_path'],    dest / 'background.png')


if __name__ == '__main__':
    input_root = Path('Preliminary-data')
    output_root = Path('Folds')
    make_stratified_folds_and_copy(input_root, output_root)
