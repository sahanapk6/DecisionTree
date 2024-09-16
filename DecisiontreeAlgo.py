
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
from sklearn import metrics

def categorize_votes(df):
    categories = {0: "no", 1: "yes", 2: "maybe"}
    categorized_df = df.replace(categories)
    return categorized_df

class DecisionTreeClassifier:
    def __init__(self):
        self.features = list
        self.X_train = np.array
        self.y_train = np.array
        self.num_feats = int
        self.train_size = int

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.features = list(X.columns)
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]

        df = X.copy()
        df['target'] = y.copy()

        # Builds Decision Tree
        self.tree = self._build_tree(df)

    def _build_tree(self, df, tree=None):
        # Get feature with maximum information gain
        feature = self._find_best_split(df)
        # Initialization of tree
        if tree is None:
            tree = {}
            tree[feature] = {}

        for feat_val in np.unique(df[feature]):
            new_df = self._split_rows(df, feature, feat_val)
            targets, count = np.unique(new_df['target'], return_counts=True)

            if (len(count) == 1):
                tree[feature][feat_val] = (targets[0], count[0])
            else:
                tree[feature][feat_val] = self._build_tree(new_df.drop(feature, axis=1))

        return tree

    def _split_rows(self, df, feature, feat_val):
        """ split rows based on given criterion """
        return df[df[feature] == feat_val].reset_index(drop=True)

    def _find_best_split(self, df):
        ig = []
        for feature in list(df.columns[:-1]):
            entropy_parent = self._get_entropy(df)
            entropy_feature_split = self._get_entropy_feature(df, feature)

            info_gain = entropy_parent - entropy_feature_split
            ig.append(info_gain)

        return df.columns[:-1][np.argmax(ig)]  # Returns feature with max information gain

    def _get_entropy(self, df):
        #Entropy of parent
        entropy = 0
        for target in np.unique(df['target']):
            fraction = df['target'].value_counts()[target] / len(df['target'])
            entropy += -fraction * np.log2(fraction)
        return entropy

    def _get_entropy_feature(self, df, feature):
        entropy = 0
        unique_feature_values = df[feature].unique()  # 0,1,2
        unique_label_values = df['target'].unique()  # 0,1

        for feature_value in unique_feature_values:
            entropy_of_label = 0
            for label_value in unique_label_values:
                num = len(df[feature][df[feature] == feature_value][df['target'] == label_value])
                den = len(df[feature][df[feature] == feature_value])

                prob = num / den
                entropy_of_label += prob * np.log2(prob)

            weight = den / len(df)
            entropy += weight * entropy_of_label
        return abs(entropy)

    def _predict_target(self, feature_lookup, x, tree, all_values=False):
        node = list(tree.keys())[0]
        if (all_values or x[node] not in tree[node]):
            res = []
            for i in tree[node].keys():
                if (type(tree[node][i]) is tuple):
                    res.append(tree[node][i])
                else:
                    node2 = tree[node][i]
                    res.append(self._predict_target(feature_lookup, x, node2, True))
            # print("checking - ", res)
            output = defaultdict(int)
            for k, v in res:
                output[k] += v
            # print("checking. - ", output)
            output = output.items()
            return max(output, key=lambda item: item[1])

        current_branch = tree[node][x[node]]
        if (type(current_branch) is tuple):
            return current_branch
        else:
            return self._predict_target(feature_lookup, x, current_branch)

    def predict(self, X):
        results = []
        feature_lookup = {key: i for i, key in enumerate(list(X.columns))}

        for index in range(len(X)):
            results.append(self._predict_target(feature_lookup, X.iloc[index], self.tree)[0])

        return np.array(results)

if __name__ == '__main__':

    warnings.simplefilter('ignore')
    data = pd.read_csv('house_votes_84.csv')
    categorized_data = categorize_votes(data)

    X = categorized_data.drop([categorized_data.columns[-1]], axis=1)
    y = categorized_data[categorized_data.columns[-1]]

    num_runs = 100
    random_seeds = range(1,100,1)
    train_accuracies = []
    test_accuracies = []

    for j in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.20,
                                                            random_state=j,
                                                            shuffle=True)

        tree = DecisionTreeClassifier()
        tree.fit(X_train, y_train)

        train_accuracy = metrics.accuracy_score(y_train, tree.predict(X_train))
        train_accuracies.append(train_accuracy)

        test_accuracy = metrics.accuracy_score(y_test, tree.predict(X_test))
        test_accuracies.append(test_accuracy)

    print(train_accuracies)
    print(test_accuracies)

    plt.hist(train_accuracies)
    plt.xlabel('accuracy on training data')
    plt.ylabel('frequency of accuracy')
    plt.show()

    plt.hist(test_accuracies)
    plt.xlabel('accuracy on test data')
    plt.ylabel('frequency of accuracy ')
    plt.show()

