import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def gather_annotated_frames(input_root: Path) -> pd.DataFrame:
    """
    Walk input_root and for each subfolder containing frame.png plus
    at least one of background.png, solid.png, or nonsolid.png,
    record the paths (None if a given mask is missing).
    """
    records = []
    for dirpath, _, filenames in os.walk(input_root):
        files = {f.lower() for f in filenames}
        # must have the frame and at least one mask
        if 'frame.png' in files and any(m in files for m in ('background.png','solid.png','non-solid.png')):
            dirp = Path(dirpath)
            rel = dirp.relative_to(input_root)
            records.append({
                'clinical_case':  rel.parts[0],
                'item':           rel.as_posix(),
                'frame_path':     dirp / 'frame.png',
                'background_path':dirp / 'background.png' if 'background.png' in files else None,
                'solid_path':     dirp / 'solid.png'      if 'solid.png'      in files else None,
                'nonsolid_path':  dirp / 'non-solid.png'   if 'non-solid.png'   in files else None,
                # optional fan:
                **({'fan_path': dirp / 'fan.png'} if 'fan.png' in files else {})
            })
    return pd.DataFrame(records)


def make_stratified_group_folds(
    input_root: Path,
    output_root: Path,
    csv_root:   Path,
    n_splits:       int = 5,
    random_state:   int = 42,
    n_bins:         int = 4
):
    df = gather_annotated_frames(input_root)

    # Build case-level metadata for stratification
    case_df = pd.DataFrame({'clinical_case': df['clinical_case'].unique()})
    hist_df = pd.read_csv(csv_root)  # needs columns: clinical_case, histological
    case_df = case_df.merge(hist_df, on='clinical_case', how='left')
    case_df['histological'].fillna('unknown', inplace=True)

    # Count frames per case and bin them
    frame_counts = df.groupby('clinical_case').size().rename('frame_count')
    case_df = case_df.join(frame_counts, on='clinical_case')
    case_df['count_bin'] = pd.qcut(
        case_df['frame_count'], q=n_bins, labels=False, duplicates='drop'
    )
    case_df['strat_label'] = (
        case_df['histological'].astype(str)
        + '_bin' + case_df['count_bin'].astype(str)
    )
    case_df.drop_duplicates(subset='clinical_case', inplace=True)

    # Prepare y and groups for StratifiedGroupKFold
    y = df['clinical_case'].map(case_df.set_index('clinical_case')['strat_label'])
    groups = df['clinical_case']
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    # Perform folds
    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(df, y=y, groups=groups)
    ):
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        print(f"=== Fold {fold_idx} ===")
        for name, split_df in (('TRAIN', train_df), ('VAL', val_df)):
            hist_prop = (
                split_df['clinical_case']
                .map(case_df.set_index('clinical_case')['histological'])
                .value_counts(normalize=True)
            )
            bin_prop = (
                split_df['clinical_case']
                .map(case_df.set_index('clinical_case')['count_bin'])
                .value_counts(normalize=True)
            )
            print(f"{name} hist proportions:\n{hist_prop.to_dict()}")
            print(f"{name} size-bin proportions:\n{bin_prop.to_dict()}\n")

        # Copy out files for this fold
        fold_dir = output_root / f'fold_{fold_idx}'
        for split_name, subset in [('train', train_df), ('val', val_df)]:
            tgt = fold_dir / split_name
            for _, row in subset.iterrows():
                dest = tgt / row['clinical_case'] / Path(row['item'])
                dest.mkdir(parents=True, exist_ok=True)

                # copy the frame
                shutil.copy2(row['frame_path'], dest / 'frame.png')

                # copy each mask if present
                if row['background_path'] is not None:
                    shutil.copy2(row['background_path'], dest / 'background.png')
                if row['solid_path'] is not None:
                    shutil.copy2(row['solid_path'], dest / 'solid.png')
                if row['nonsolid_path'] is not None:
                    shutil.copy2(row['nonsolid_path'], dest / 'non-solid.png')

                # optional fan
                if 'fan_path' in row and pd.notnull(row['fan_path']):
                    shutil.copy2(row['fan_path'], dest / 'fan.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root',  type=Path, default=Path('Preliminary-data'))
    parser.add_argument('--output_root', type=Path, default=Path('Multiclass_Folds'))
    parser.add_argument('--csv_root',    type=Path, default=Path('clinical-cases-metadata-holsbeke.csv'))
    parser.add_argument('--splits',      type=int,  default=5)
    parser.add_argument('--seed',        type=int,  default=42)
    args = parser.parse_args()

    make_stratified_group_folds(
        input_root   = args.input_root,
        output_root  = args.output_root,
        csv_root     = args.csv_root,
        n_splits     = args.splits,
        random_state = args.seed,
        n_bins       = 4
    )
