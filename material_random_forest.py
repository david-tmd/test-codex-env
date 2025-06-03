import random
from typing import List, Tuple


def load_material_data() -> Tuple[List[List[float]], List[float]]:
    """Return toy data for material property prediction.

    Each sample has the following features:
    [density (g/cm^3), melting point (C), atomic number].
    The target represents an arbitrary hardness value.
    """
    data = [
        [2.7, 660.3, 13, 2.75],  # Aluminum
        [7.9, 1538.0, 26, 4.0],  # Iron
        [8.9, 1085.0, 29, 3.0],  # Copper
        [19.3, 3422.0, 74, 4.5],  # Tungsten
        [7.1, 232.0, 50, 1.5],   # Tin
    ]
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    return X, y


class DecisionTree:
    def __init__(self, max_depth: int = 3, feature_subset_size: int = 1):
        self.max_depth = max_depth
        self.feature_subset_size = feature_subset_size
        self.tree = None

    def fit(self, X: List[List[float]], y: List[float]):
        data = list(zip(X, y))
        features = list(range(len(X[0])))
        self.tree = self._build_tree(data, depth=0, features=features)

    def _build_tree(self, data: List[Tuple[List[float], float]], depth: int, features: List[int]):
        ys = [target for _, target in data]
        if depth >= self.max_depth or len(set(ys)) == 1:
            return sum(ys) / len(ys)

        chosen_features = random.sample(features, self.feature_subset_size)
        best_feature, best_threshold, best_score = None, None, float('inf')

        for feature in chosen_features:
            values = sorted(set(row[feature] for row, _ in data))
            for threshold in values:
                left = [target for row, target in data if row[feature] <= threshold]
                right = [target for row, target in data if row[feature] > threshold]
                if not left or not right:
                    continue
                left_mean = sum(left) / len(left)
                right_mean = sum(right) / len(right)
                score = sum((t - left_mean) ** 2 for t in left) + sum((t - right_mean) ** 2 for t in right)
                if score < best_score:
                    best_feature = feature
                    best_threshold = threshold
                    best_score = score

        if best_feature is None:
            return sum(ys) / len(ys)

        left_data = [(row, target) for row, target in data if row[best_feature] <= best_threshold]
        right_data = [(row, target) for row, target in data if row[best_feature] > best_threshold]

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(left_data, depth + 1, features),
            'right': self._build_tree(right_data, depth + 1, features),
        }

    def _predict_one(self, row: List[float]):
        node = self.tree
        while isinstance(node, dict):
            if row[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self._predict_one(row) for row in X]


class RandomForest:
    def __init__(self, n_estimators: int = 10, max_depth: int = 3, feature_subset_size: int = 1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subset_size = feature_subset_size
        self.trees: List[DecisionTree] = []

    def fit(self, X: List[List[float]], y: List[float]):
        self.trees = []
        for _ in range(self.n_estimators):
            indices = [random.randrange(len(X)) for _ in range(len(X))]
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]
            tree = DecisionTree(max_depth=self.max_depth, feature_subset_size=self.feature_subset_size)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: List[List[float]]) -> List[float]:
        predictions = []
        for row in X:
            preds = [tree._predict_one(row) for tree in self.trees]
            predictions.append(sum(preds) / len(preds))
        return predictions


def main():
    X, y = load_material_data()
    model = RandomForest(n_estimators=5, max_depth=4, feature_subset_size=2)
    model.fit(X, y)
    preds = model.predict(X)
    for features, pred in zip(X, preds):
        print(f"Features {features} -> predicted hardness {pred:.2f}")


if __name__ == "__main__":
    main()
