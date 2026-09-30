"""
Unsupervised Anomaly Isolation Forest.
Pure NumPy implementation for outlier detection on legitimate transactions.
"""
from __future__ import annotations
import numpy as np


def c_factor(n: int) -> float:
    """Average path length of unsuccessful searches in a binary search tree (Euler constant approximation)."""
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.0
    euler_gamma = 0.57721566490153286
    return 2.0 * (np.log(n - 1.0) + euler_gamma) - (2.0 * (n - 1.0) / n)


class IsolationNode:
    def __init__(
        self,
        feature_idx: int | None = None,
        split_value: float | None = None,
        left: "IsolationNode | None" = None,
        right: "IsolationNode | None" = None,
        size: int = 0,
    ):
        self.feature_idx = feature_idx
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = size

    @property
    def is_leaf(self) -> bool:
        return self.feature_idx is None


class IsolationTree:
    def __init__(self, max_depth: int, random_state: int | None = None):
        self.max_depth = max_depth
        self.root = None
        self.rng = np.random.RandomState(random_state)

    def _build_tree(self, X: np.ndarray, current_depth: int = 0) -> IsolationNode:
        n_samples, n_features = X.shape
        if current_depth >= self.max_depth or n_samples <= 1:
            return IsolationNode(size=n_samples)

        feat = self.rng.randint(0, n_features)
        feat_min = X[:, feat].min()
        feat_max = X[:, feat].max()

        if feat_min == feat_max:
            return IsolationNode(size=n_samples)

        split_val = self.rng.uniform(feat_min, feat_max)
        left_mask = X[:, feat] < split_val
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return IsolationNode(size=n_samples)

        left_node = self._build_tree(X[left_mask], current_depth + 1)
        right_node = self._build_tree(X[right_mask], current_depth + 1)

        return IsolationNode(
            feature_idx=feat, split_value=split_val, left=left_node, right=right_node, size=n_samples
        )

    def fit(self, X: np.ndarray) -> "IsolationTree":
        self.root = self._build_tree(X, current_depth=0)
        return self

    def _path_length(self, x: np.ndarray, node: IsolationNode, current_depth: int = 0) -> float:
        if node.is_leaf:
            return current_depth + (c_factor(node.size) if node.size > 1 else 0.0)
        if x[node.feature_idx] < node.split_value:
            return self._path_length(x, node.left, current_depth + 1)
        return self._path_length(x, node.right, current_depth + 1)

    def path_lengths(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._path_length(row, self.root, 0) for row in X], dtype=np.float64)


class PureIsolationForest:
    """Pure NumPy Isolation Forest."""
    def __init__(
        self,
        n_estimators: int = 40,
        max_samples: int = 256,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.trees: list[IsolationTree] = []
        self.rng = np.random.RandomState(random_state)
        self.c_val = 1.0

    def fit(self, X: np.ndarray) -> "PureIsolationForest":
        n_samples = X.shape[0]
        subsample_size = min(self.max_samples, n_samples)
        max_depth = int(np.ceil(np.log2(max(subsample_size, 2))))
        self.c_val = c_factor(subsample_size)
        self.trees = []

        for i in range(self.n_estimators):
            seed = self.rng.randint(0, 1_000_000)
            tree_rng = np.random.RandomState(seed)
            sub_idx = tree_rng.choice(n_samples, size=subsample_size, replace=False)
            tree = IsolationTree(max_depth=max_depth, random_state=seed)
            tree.fit(X[sub_idx])
            self.trees.append(tree)

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns raw anomaly score: higher = more anomalous."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        lengths = np.array([tree.path_lengths(X) for tree in self.trees])
        avg_len = np.mean(lengths, axis=0)
        # Score s = 2 ^ (-avg_len / c(n)) -> anomalies have short avg_len, so score -> 1.0
        score = np.power(2.0, -avg_len / max(self.c_val, 1e-12))
        return score


def train_isolation_forest(X_legit_scaled: np.ndarray) -> PureIsolationForest:
    """Train Isolation Forest on scaled legitimate rows."""
    model = PureIsolationForest(n_estimators=40, max_samples=256, random_state=42)
    model.fit(X_legit_scaled)
    return model


def raw_anomaly(model: PureIsolationForest, X_scaled: np.ndarray) -> np.ndarray:
    return model.score_samples(X_scaled)


def fit_score_range(model: PureIsolationForest, X_scaled: np.ndarray) -> tuple[float, float]:
    raw = raw_anomaly(model, X_scaled)
    return float(np.min(raw)), float(np.max(raw))


def anomaly_score(model: PureIsolationForest, X_scaled: np.ndarray, lo: float, hi: float) -> np.ndarray:
    raw = raw_anomaly(model, X_scaled)
    span = (hi - lo) + 1e-12
    return np.clip((raw - lo) / span, 0.0, 1.0)
