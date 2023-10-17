from decimal import Decimal
from pandas import DataFrame, Series
from prettytable import prettytable
from modules.classes import DTree
from modules.constants import LEAVES, TREE_TYPES
from training_allheart_old import Criteria


def bin_target_column(df, boolean_expression_variable: int,
                      boolean_expression_operation: str = "greater_than_eq") -> tuple:
    """
    Performs binning on the target column, converting it from an integer column into a
    binary column.

    :return: The original dataframe with the transformed column.
    """
    # target_binned_name = f"target_binary"
    binning_dict = {
        True: 1,
        False: 0
    }
    input_columns = df.iloc[:, 0:-1]
    target_column = df.iloc[:, -1]

    if boolean_expression_operation == "greater_than_eq":
        bool_target = target_column >= boolean_expression_variable
        df["target"] = bool_target.map(binning_dict)
        binary_target_column = bool_target.map(binning_dict)
    else:
        raise ValueError(
            f"Method does not have instructions for handling the given boolean operation,"
            f" {boolean_expression_operation!r}")

    return df, input_columns, binary_target_column


def modeling(input_data: DataFrame, target_data: Series, model_settings: dict):
    """
    Function to model and print decision tree information.

    :param input_data: The input dataframe.
    :param target_data: The target series.
    :param model_settings:

    :return: The DTree object containing the decision tree information.
    """
    training_partition, split_type, leaf_size, number_of_model = model_settings.values()
    model_settings["testing_partition"] = 1 - training_partition
    training_partition_perc = int(training_partition * 100)
    testing_partition_perc = int((1 - training_partition) * 100)
    model_name = f"model-{number_of_model}"
    testing = DTree(input_data, target_data, model_settings)
    print(f"{model_name} has been trained.")
    # testing.export_tree(model_name)
    testing.features(model_name)
    return testing


def table_data(tree_array):
    """
    Generates the lists of evaluation metrics for the table generation of the decision
    trees.

    :param tree_array: List of decision tree objects.
    :return: Tuple containing the accuracy, simplicity, lift, stability, and overall
    scores.
    """
    accuracy = []
    simplicity = []
    lift = []
    stability = []
    overall = []
    precision = lambda val: float(Decimal(val).quantize(Decimal("0.00001")))

    for tree in tree_array:
        obj = Criteria(tree)
        a_raw = precision(obj.accuracy)
        a_score = obj.criteria["accuracy"]
        pair = [a_raw, a_score]
        accuracy.append(pair)

        si_leaves = obj.leaves
        si_score = obj.criteria["simplicity"]
        pair = [si_leaves, si_score]
        simplicity.append(pair)

        l_tree, l_best = obj.tree_matrix.loc[2, ["cum_resp_pct", "cum_resp_pct_wiz"]]
        l_score = obj.criteria["lift"]
        trio = [l_tree, l_best, l_score]
        lift.append(trio)

        st_score = obj.criteria["stability"]
        if st_score:
            st_just = "stable"
        else:
            st_just = "unstable"
        pair = [st_score, st_just]
        stability.append(pair)

        o_score = obj.score
        overall.append(o_score)

    return accuracy, simplicity, lift, stability, overall


def print_table(accuracy, simplicity, lift, stability, overall):
    """
    Prints a table displaying the evaluation metrics for the four decision trees.

    :param accuracy: Accuracy scores as a list of lists [raw, score].
    :param simplicity: Simplicity scores as a list of lists [leaf count, score].
    :param lift: Lift scores as a list of lists [tree, best, score].
    :param stability: Stability scores as a list of lists [score, justification].
    :param overall: Overall scores as a list.
    :return: A PrettyTable object representing the table.
    """
    ptable = prettytable.PrettyTable()

    ptable.add_column("Tree\n[\"Tree Name\"]", ["Tree 1", "Tree 2", "Tree 3", "Tree 4"])
    ptable.add_column("Accuracy\n[\"Raw\", \"Score\"]", [accuracy[0], accuracy[1],
                                                         accuracy[2], accuracy[3]])
    ptable.add_column("Simplicity\n[\"Leaf Count\", \"Score\"]",
                      [simplicity[0], simplicity[1], simplicity[2], simplicity[3]])
    ptable.add_column("Lift\n[\"Tree\", \"Best\", \"Score\"]", [lift[0], lift[1],
                                                                lift[2],
                                                                lift[3]])
    ptable.add_column("Stability\n[\"Score\", \"Justification\"]",
                      [stability[0], stability[1], stability[2], stability[3]])
    ptable.add_column("Overall Score\n[\"Score Sum\"]", [overall[0], overall[1], overall[2], overall[3]])

    ptable.reversesort = True
    ptable.sortby = "Overall Score\n[\"Score Sum\"]"

    return ptable
