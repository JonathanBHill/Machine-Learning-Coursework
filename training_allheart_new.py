# from modules.classes import
import pandas as pd

from modules.classes import Pckg
from modules.constants import LEAVES, TREE_TYPES
from modules.functions import bin_target_column, modeling


def main():
    package = Pckg()
    decision_trees = []
    count = 0
    base_df = pd.read_csv(f"{package.csv_path}allheart.csv")

    for col in base_df:
        median = base_df[col].median()
        base_df[col].fillna(median, inplace=True)

    binned_target_df, input_columns, target_column = bin_target_column(base_df, 1)

    for leaf in LEAVES:
        for tree_type in TREE_TYPES:
            count += 1
            settings_for_model = {
                "training_partition": 0.7,
                "split_type": tree_type,
                "leaf_size": leaf,
                "model_number": str(count)
            }
            decision_tree = modeling(input_columns, target_column, settings_for_model)

            decision_trees.append(decision_tree)
    return decision_trees


if __name__ == '__main__':
    main()
