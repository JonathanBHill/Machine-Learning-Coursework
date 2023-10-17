from decimal import Decimal
from pathlib import Path

import kds
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster, model_selection, preprocessing, tree as sktree
from modules.constants import LEAVES, RANDOM_STATE, TREE_TYPES

class Pckg:
    def __init__(self):
        self.path = Path()
        self.modules_path = self.path.cwd()
        self.package_path, self.package_name = self.package_location()
        self.notebooks_path = f"{self.package_path}/notebooks/"
        self.data_path = f"{self.package_path}/data/"
        self.csv_path = f"{self.data_path}csv/"
        self.dot_path = f"{self.data_path}dot/"
        self.png_path = f"{self.data_path}png/"
        self.directory_paths = {}

    def package_location(self):
        proj_list = self.path.cwd().as_posix().split("/")
        proj_list = proj_list[:5]
        return "/".join(proj_list), proj_list[-1]


class DTree:
    def __init__(self, inputs, output, settings: dict):
        """
        Initializes the DTree class.

        :param inputs: The columns used as inputs for the decision tree.
        :param output: The column used as the target for the decision tree.
        :param leaf: Leaf size as a percentage of the training set.
        """
        self.in_out_data = [inputs, output]
        self._ptraining = settings["training_partition"]
        self._ptesting = settings["testing_partition"]
        self._leaf = settings["leaf_size"]
        self._criterion = settings["split_type"]
        self._xtrain, self._xtest, self._ytrain, self._ytest = self._split_data()
        self._leafsize = self._get_size_of_leaf()
        self.shape = {
            "training": self._xtrain.shape,
            "testing": self._ytest.shape
        }
        self.xydata = {
            "all": [self._xtest, self._ytest, self._xtrain, self._ytrain],
            "tests": [self._xtest, self._ytest],
            "training": [self._xtrain, self._ytrain],
            "x": [self._xtest, self._xtrain],
            "y": [self._ytest, self._ytrain]
        }
        self.tree = sktree.DecisionTreeClassifier(
            criterion=self._criterion,
            random_state=RANDOM_STATE,  # 100
            min_samples_leaf=self._leafsize
        )
        self.ccp_alphas, self.impurities = self.tree.cost_complexity_pruning_path(
            self._xtrain, self._ytrain).values()
        self._fitted_dtree = self._create_fitted_decision_tree()
        self._features = self._fitted_dtree.feature_importances_
        self.package = Pckg()

    def _get_size_of_leaf(self):
        """
        Calculates the leaf size for the decision tree.

        :return: The calculated leaf size.
        """
        # test = self._xtrain.shape[0] * self._ptraining * self._leaf / 100
        # print(test)
        size = self._xtrain.shape[0] * self._ptraining * self._leaf / 100

        return int(round(size))

    def _split_data(self):
        """
        Splits the data into training and testing sets.

        :return: The split data as (xtrain, xtest, ytrain, ytest).
        """
        distribution = model_selection.train_test_split(
            self.in_out_data[0], self.in_out_data[1],
            test_size=self._ptesting, stratify=self.in_out_data[1]
        )
        return distribution

    def _create_fitted_decision_tree(self):
        """
        Creates a decision tree and fits it to the training data.

        :return: The fitted decision tree.
        """
        dtree = sktree.DecisionTreeClassifier(
            criterion=self._criterion,
            random_state=RANDOM_STATE,
            min_samples_leaf=self._leafsize
        )

        dtree_fit = dtree.fit(self._xtrain, self._ytrain)

        return dtree_fit

    def export_tree(self, filename: str):
        """
        Exports the decision tree to a .dot file.

        :param filename: The name of the output .dot file.
        """
        filename = f"{self.package.dot_path}{filename}.dot"
        sktree.export_graphviz(self._fitted_dtree, out_file=filename)
        print(f"{filename!r} exported to {self.package.dot_path}.")

    def features(self, filename: str = None):
        """
        Shows the importance of each input on the target.

        :param filename: The name of the output .png file.
        """
        filename = f"{self.package.png_path}{filename}.png"

        features = self._fitted_dtree.feature_importances_
        varz = self.in_out_data[0].columns

        for feature in features:
            feature = Decimal(feature).quantize(Decimal("0.00001"))

        plt.bar([x for x in varz], features)
        plt.xticks(rotation=60)
        plt.title("Feature Importances of Inputs on Charges")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _shape(self) -> dict[str, int]:
        """
        Finds the shape of the training and testing dataset split.

        :return: The shape of the training and testing datasets.
        """
        shapes = {
            "training": self._xtrain.shape,
            "testing": self._ytest.shape
        }
        return shapes

    def _update(self):
        """Reruns the shape method to update the shape criteria."""
        self.shape = self._shape()


class Criteria:
    def __init__(self, decision_tree: DTree):
        """
        Initializes the Criteria class.

        :param decision_tree: The decision tree object of type DTree.
        """
        self.tree_obj = decision_tree
        self.tree = self.tree_obj.fitted_dtree
        self.weights = np.array([0.55, 0.25, 0.1, 0.1], dtype=float)
        self.xe, self.ye, self.xt, self.yt = self.tree_obj.xydata["all"]
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
