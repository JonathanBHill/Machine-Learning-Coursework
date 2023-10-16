#%% Imports
from _decimal import Decimal
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd
import numpy as np
import sklearn.cluster as skcluster
import sklearn.model_selection as skmodels
import sklearn.preprocessing as skpreproc
import sklearn.tree as sktrees
import prettytable
import kds

#%% Classes
class DataPreparation:
    def __init__(self, dataframe: DataFrame):
        """
        Initializes the DataPreparation class.

        :param dataframe: The initial DataFrame to be processed.
        """
        self.df = dataframe.copy(True)
        self.columns = dataframe.columns.tolist()
        self.dfimp = self._imputation(self.df.copy(True))

    def _findnull(self):
        """
        Finds and returns information about null values in the DataFrame.

        :return: DataFrame with columns indicating the presence of null values and their count.
        """
        cols_names = []
        lower_cols = self.columns

        for index in range(0, len(lower_cols)):
            col = lower_cols[index].capitalize()
            cols_names.append(col)
        nulls_counts = list(zip(self.df.isnull().any(), self.df.isnull().sum()))

        df = pd.DataFrame(
            data=nulls_counts,
            index=cols_names,
            columns=["Boolean", "Count"]
        )

        return df

    def _imputation(self, df) -> DataFrame:
        """
        Performs imputation on null values in the DataFrame using the median.

        :param df: The DataFrame to perform imputation on.
        :return: DataFrame with imputed values for nulls.
        """
        for column in df:
            test = df[column].isnull().any()
            if test:
                raw = df[column]
                sorted_col = raw.sort_values()
                sorted_col = sorted_col.dropna()
                col_size = sorted_col.size
                col_even = bool(col_size % 2 == 0)
                median = self._median(col_even, sorted_col)
                df[column] = df[column].fillna(median)

        return df

    @staticmethod
    def _median(even, data) -> float:
        """
        Calculates the median for a specified dataset.

        :param even: Specifies whether the dataset size is even or odd.
        :param data: DataFrame containing the data to analyze.
        :return: The calculated median.
        """
        if even:
            half = int(data.size / 2)
        else:
            half = int((data.size - 1) / 2)

        median_row = half + 1
        median = data[median_row]

        return median

class DataModeling:
    def __init__(self, dataframe, clusters):
        """
        Initializes the DataModeling class.

        :param dataframe: The dataframe used for data modeling.
        :param clusters: The number of clusters for creating the model.
        """
        self.df = dataframe
        self.clustering_df = pd.DataFrame()
        self._clusters = clusters
        self._cluster_labels = range(0, clusters, 1)
        self.input_cols = dataframe.iloc[:, 0:-1]
        self.target_col = dataframe.iloc[:, -1]
        self._data: dict = {
            "x": self.input_cols,
            "y": self.target_col
        }
        self._kmeans = None
        self._kmeans_kwargs = dict(
            init="random", n_init="auto", max_iter=300, random_state=42
        )

    def target_binning(self, column) -> DataFrame:
        """
        Performs binning on the target column, converting it from an integer column into a binary column.

        :param column: The name of the column to be transformed.
        :return: The original dataframe with the transformed column.
        """
        clustered_col = f"bi_{column}"

        self.df[clustered_col] = self.df[column] >= 1
        self.df[clustered_col] = self.df[clustered_col].astype(int)
        self.clustering_df = self.df.copy(True)

        return self.clustering_df

    def sse(self, display=False):
        """
        Calculates the sum of squared errors (SSE) for different numbers of clusters.

        :param display: Flag to say whether to display the SSE-Number Plot Chart.
        :return: A list of SSE values for different numbers of clusters.
        """
        scaler = skpreproc.StandardScaler()
        scaled_features = scaler.fit_transform(self._data["x"])
        kkw = self._kmeans_kwargs
        sse = []

        for k in range(1, 11):
            kmeans_x = skcluster.KMeans(n_clusters=k, **kkw)
            kmeans_x.fit(scaled_features)
            sse.append(kmeans_x.inertia_)

        plt.style.use('fivethirtyeight')
        plt.plot(range(1, 11), sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.title("SSE-Number Plot Chart")
        plt.tight_layout()
        if display:
            # plt.show()
            plt.savefig("sse.png")
        plt.close()

        return sse

class DTree:
    def __init__(self, inputs, output, leaf):
        """
        Initializes the DTree class.

        :param inputs: The columns used as inputs for the decision tree.
        :param output: The column used as the target for the decision tree.
        :param leaf: Leaf size as a percentage of the training set.
        """
        self._input_data = inputs
        self._output_data = output
        self.in_out_data = [inputs, output]
        self._ptraining = 0.8
        self._ptesting = 1 - self._ptraining
        self._leaf = leaf
        self._criterion = "gini"
        self._xtrain, self._xtest, self._ytrain, self._ytest = self.__split_data()
        self._leafsize = self.__leaf()

        self.shape = self.__shape()
        self.tree = sktrees.DecisionTreeClassifier(
            criterion=self._criterion,
            random_state=100,
            min_samples_leaf=self._leafsize
        )
        self._dtree_fit = self.tree.fit(self._xtrain, self._ytrain)
        self._features = self._dtree_fit.feature_importances_

    def xydata(self, rtn_type: str = "all"):
        """
        Returns x and y data after splitting property.

        :param rtn_type: Type of datasets to return ("tests", "training", "x", "y", "all").
        :return: The requested data based on the specified return type.
        """
        data = []
        if rtn_type == "tests":
            data = [self._xtest, self._ytest]
        elif rtn_type == "training":
            data = [self._xtrain, self._ytrain]
        elif rtn_type == "x":
            data = [self._xtest, self._xtrain]
        elif rtn_type == "y":
            data = [self._ytest, self._ytrain]
        elif rtn_type == "all":
            data = [self._xtest, self._ytest, self._xtrain, self._ytrain]

        return data

    @property
    def partition(self):
        """
        Returns the partition property.

        :return: The partition property as a dictionary with keys "training" and "testing".
        """
        partition = {
            "training": self._ptraining,
            "testing": self._ptesting
        }
        return partition

    @partition.setter
    def partition(self, percent: float):
        """
        Sets the partition properties.

        :param percent: The percentage for the training set.
        """
        if percent > 1 or percent < 0:
            raise ValueError("Value must be between 0 and 1.")
        else:
            self._ptraining = percent
            self._ptesting = 1 - percent
        self.__update()

    @property
    def criterion(self):
        """
        Returns the criterion property.

        :return: The criterion property.
        """
        return self._criterion

    @criterion.setter
    def criterion(self, method: str = "gini"):
        """
        Sets the criterion property.

        :param method: The criterion method to use.
        """
        self._criterion = method

    @property
    def dtree(self):
        """
        Returns the dtree property.

        :return: The dtree property.
        """
        return self._dtree_fit

    @dtree.setter
    def dtree(self, tree):
        """
        Sets the dtree property.

        :param tree: The decision tree to set.
        """
        print(tree)
        self._dtree_fit = tree

    def export_tree(self, filename: str):
        """
        Exports the decision tree to a .dot file.

        :param filename: The name of the output .dot file.
        """
        filename = f"{filename}.dot"
        sktrees.export_graphviz(self._dtree_fit, out_file=filename)

    def _decision_tree(self):
        """
        Creates a decision tree and fits it to the training data.

        :return: The fitted decision tree.
        """
        dtree = sktrees.DecisionTreeClassifier(
            criterion=self._criterion,
            random_state=100,
            min_samples_leaf=self._leafsize
        )

        dtree_fit = dtree.fit(self._xtrain, self._ytrain)

        return dtree_fit

    def features(self, filename: str):
        """
        Shows the importance of each input on the output.

        :param filename: The name of the output .png file.
        """
        filename = f"{filename}.png"

        feat = self._dtree_fit.feature_importances_
        varz = self._input_data.columns

        for i, s in enumerate(feat):
            s = Decimal(s).quantize(Decimal("0.00001"))
            print(f"Feature: {varz[i]}; Score: {s}")

        plt.bar([x for x in varz], feat)
        plt.xticks(rotation=60)
        plt.title("Feature Importances of Inputs on Charges")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def __leaf(self):
        """
        Calculates the leaf size for the decision tree.

        :return: The calculated leaf size.
        """
        size = self._xtrain.shape[0] * self._ptraining * self._leaf / 100

        return int(round(size))

    def __split_data(self):
        """
        Splits the data into training and testing sets.

        :return: The split data as (xtrain, xtest, ytrain, ytest).
        """
        distribution = skmodels.train_test_split(
            self._input_data, self._output_data,
            test_size=self._ptesting, stratify=self._output_data
        )
        return distribution

    def __shape(self):
        """
        Finds the shape of the training and testing dataset split.

        :return: The shape of the training and testing datasets.
        """
        shapes = {
            "training": self._xtrain.shape,
            "testing": self._ytest.shape
        }
        return shapes

    def __update(self):
        """Reruns the shape method to update the shape criteria."""
        self.shape = self.__shape()

class Criteria:
    def __init__(self, decision_tree: DTree):
        """
        Initializes the Criteria class.

        :param decision_tree: The decision tree object of type DTree.
        """
        self.tree_obj = decision_tree
        self.tree = self.tree_obj.dtree
        self.xe = self.tree_obj.xydata("all")
        self.weights = np.array([0.55, 0.25, 0.1, 0.1], dtype=float)
        self.xe, self.ye, self.xt, self.yt = self.tree_obj.xydata("all")
        self.leaves = self.tree.get_n_leaves()
        self.accuracy = self.tree.score(self.xe, self.ye)
        self.tree_matrix = None
        self.simplicity = self.score_simplicity()
        self.lift = self.score_lift()
        self.stability = self.score_stability()
        self.criteria = self.tree_criteria()
        self.score = float(Decimal(sum(self.criteria.values())).quantize((Decimal("0.00001"))))

    def show_stability(self):
        """
        Displays a scatter plot of the stability score based on the decile values in the tree matrix.
        """
        x_axis = self.tree_matrix["decile"].values
        y_axis = self.tree_matrix["cnt_resp"].values
        plt.scatter(x_axis, y_axis)
        plt.plot(x_axis, y_axis, "--r")
        plt.show()

    def score_simplicity(self):
        """
        Calculates the simplicity score based on the number of leaves in the decision tree.

        :return: The simplicity score.
        """
        leaves = self.leaves
        simplicity = None

        if leaves <= 2 or leaves >= 13:
            simplicity = 0
        elif 3 <= leaves <= 4:
            simplicity = (leaves - 2) / 3
        elif 5 <= leaves <= 8:
            simplicity = 1
        elif 9 <= leaves <= 12:
            simplicity = (13 - leaves) / 5

        return simplicity

    def score_lift(self):
        """
        Calculates the lift score based on the prediction probabilities of the decision tree.

        :return: The lift score.
        """
        y_probability = self.tree.predict_proba(self.xe)
        self.tree_matrix = kds.metrics.decile_table(self.ye.values, y_probability[:, 1])
        cs_responders = self.tree_matrix.loc[2,["cum_resp_pct"]].values[0]
        cs_responders_best = self.tree_matrix.loc[2,["cum_resp_pct_wiz"]].values[0]
        lift = round((cs_responders - 30) / (cs_responders_best - 30), ndigits=4)
        return lift

    def score_stability(self):
        """
        Calculates the stability score based on the decile values in the tree matrix.

        :return: The stability score.
        """
        stability = None
        x_axis = self.tree_matrix["decile"].values
        y_axis = self.tree_matrix["cnt_resp"].values
        second_decile = y_axis[1]
        third_decile = y_axis[2]
        if second_decile > third_decile:
            stability = 1
        else:
            stability = 0

        return stability

    def tree_criteria(self):
        """
        Calculates the criteria scores for accuracy, simplicity, lift, and stability.

        :return: A dictionary containing the criteria scores.
        """
        scores = np.array(
            [self.accuracy, self.simplicity, self.lift, self.stability],
            dtype=float
        )
        scores = scores * self.weights
        precision = lambda score : float(Decimal(score).quantize(Decimal("0.00001")))
        criteria = {
            "accuracy": precision(scores[0]),
            "simplicity": precision(scores[1]),
            "lift": precision(scores[2]),
            "stability": precision(scores[3])
        }

        return criteria

#%% Functions
def modeling(df: DataFrame, training_partition: float,
             splitting: str, leaf_size: float, model: str = "1"):
    """
    Function to model and print decision tree information.

    :param df: The input dataframe containing the data.
    :param training_partition: The proportion of data to be used for training (between 0 and 1).
    :param splitting: The type of splitting to be used in the decision tree (e.g., "gini" or "entropy").
    :param leaf_size: The size of leaves as a percentage of the training set.
    :param model: The suffix used for the model's file name.
    :return: The DTree object containing the decision tree information.
    """
    testing_partition = 1 - training_partition
    testing_partition = Decimal(testing_partition).quantize(Decimal("0.1"))
    testx = df.iloc[:, 0:-1]
    columns = testx.columns.tolist()
    testy = df.iloc[:, -1]
    model = f"model-{model}"
    print(f"Model: {model}:")
    print(
        f"Columns used:\n{pd.Series(columns, name='Columns')}\n"
        f"\nPartitioning: {int(training_partition * 100)}% training"
        f"/{int(testing_partition * 100)}% testing\n"
        f"Splitting type: {splitting.capitalize()}\n"
        f"Leaf size: {leaf_size}% of training set\n"
        )
    testing = DTree(testx, testy, leaf_size)
    testing.partition = training_partition
    testing.criterion = splitting
    testing.export_tree(model)
    testing.features(model)
    return testing

def table_data(tree_array):
    """
    Generates the lists of evaluation metrics for the table generation of the decision trees.

    :param tree_array: List of decision tree objects.
    :return: Tuple containing the accuracy, simplicity, lift, stability, and overall scores.
    """
    accuracy = []
    simplicity = []
    lift = []
    stability = []
    overall = []
    precision = lambda val : float(Decimal(val).quantize(Decimal("0.00001")))

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

        l_tree, l_best = obj.tree_matrix.loc[2,["cum_resp_pct", "cum_resp_pct_wiz"]]
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

def print_table(a,  si, l, st, o):
    """
    Prints a table displaying the evaluation metrics for the four decision trees.

    :param a: Accuracy scores as a list of lists [raw, score].
    :param si: Simplicity scores as a list of lists [leaf count, score].
    :param l: Lift scores as a list of lists [tree, best, score].
    :param st: Stability scores as a list of lists [score, justification].
    :param o: Overall scores as a list.
    :return: A PrettyTable object representing the table.
    """
    ptable = prettytable.PrettyTable()

    ptable.add_column("Tree\n[\"Tree Name\"]", ["Tree 1","Tree 2","Tree 3","Tree 4"])
    ptable.add_column("Accuracy\n[\"Raw\", \"Score\"]", [a[0],a[1],a[2],a[3]])
    ptable.add_column("Simplicity\n[\"Leaf Count\", \"Score\"]", [si[0],si[1],si[2],si[3]])
    ptable.add_column("Lift\n[\"Tree\", \"Best\", \"Score\"]", [l[0],l[1],l[2],l[3]])
    ptable.add_column("Stability\n[\"Score\", \"Justification\"]", [st[0],st[1],st[2],st[3]])
    ptable.add_column("Overall Score\n[\"Score Sum\"]", [o[0],o[1],o[2],o[3]])

    ptable.reversesort = True
    ptable.sortby = "Overall Score\n[\"Score Sum\"]"

    return ptable

#%% Main Program
def main():
    base_df = pd.read_csv("data/csv/allheart.csv")
    prep_df = DataPreparation(base_df)
    cleaned_df = prep_df.dfimp
    model = DataModeling(cleaned_df, 2)
    binned_target_df = model.target_binning("target")

    leaf = [3.5, 6]
    tree_type = ["entropy", "gini"]
    binned_target_df = binned_target_df.drop(columns="target")

    d_tree1 = modeling(binned_target_df, 0.7, tree_type[0], leaf[0], "I")
    d_tree2 = modeling(binned_target_df, 0.7, tree_type[1], leaf[0], "II")
    d_tree3 = modeling(binned_target_df, 0.7, tree_type[0], leaf[1], "III")
    d_tree4 = modeling(binned_target_df, 0.7, tree_type[1], leaf[1], "IV")

    trees = [d_tree1, d_tree2, d_tree3, d_tree4]
    accuracy, simplicity, lift, stability, overall = table_data(trees)
    tbl = print_table(accuracy, simplicity, lift, stability, overall)

    leaf = [3.5, 6]
    tree_type = ["entropy", "gini"]
    dclass = modeling(binned_target_df, 0.7, tree_type[0], leaf[0], "None")
    dtesting = sktrees.DecisionTreeClassifier(
        random_state=100,
        min_samples_leaf=34
    )
    xt, yt = dclass.xydata("training")
    xt = xt.iloc[:, 0:13]
    path = dtesting.cost_complexity_pruning_path(xt, yt)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fitted = dtesting.fit(xt, yt)
    dtesting.get_n_leaves()
    sktrees.export_graphviz(fitted, out_file="testing_dot")

    clf = sktrees.DecisionTreeClassifier(random_state=100, ccp_alpha=ccp_alphas[-1])
    clf.fit(xt, yt)
    sktrees.export_graphviz(fitted, out_file="testing_two")

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = sktrees.DecisionTreeClassifier(random_state=100, ccp_alpha=ccp_alpha)
        clf.fit(xt, yt)
        clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()


if __name__ == "__main__":
    main()

