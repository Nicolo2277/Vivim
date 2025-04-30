import pandas as pd
from sklearn.model_selection import StratifiedKFold

def make_stratified_group_folds(metadata_path, train_items_path, n_splits=5, random_state=42):
    df_histological = pd.read_csv(metadata_path)

    df_train_items = pd.read_csv(train_items_path)

    common = set(df_histological["item"]) & set(df_train_items["item"])
    df = df_histological[df_histological["item"].isin(common)].reset_index(drop=True)



    case_labels = df.groupby("clinical_case")["histological"] \
                    .agg(lambda s: s.mode().iat[0]).reset_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []

    for train_idx, val_idx in skf.split(case_labels["clinical_case"], case_labels["histological"]):
        train_cases = case_labels.loc[train_idx, "clinical_case"].tolist()
        val_cases  = case_labels.loc[val_idx,  "clinical_case"].tolist()
        folds.append((train_cases, val_cases))
    
    return df, folds
        
    

        


