"""
Supervised Fraud Detection Balanced Random Forest Classifier.
Pure NumPy implementation optimized for imbalanced fraud classification.
"""
from __future__ import annotations
import numpy as np


class DecisionNode:
    def __init__(
        self,
        feature_idx: int | None = None,
        threshold: float | None = None,
        left: "DecisionNode | None" = None,
        right: "DecisionNode | None" = None,
        prob: float | None = None,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prob = prob  # Probability of class 1 (Fraud)

    @property
    def is_leaf(self) -> bool:
        return self.prob is not None


class FastDecisionTree:
    """Fast Binary Decision Tree with Gini criterion."""
    def __init__(
        self,
        max_depth: int = 7,
        min_samples_split: int = 8,
        max_features: int | str = "sqrt",
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.rng = np.random.RandomState(random_state)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int | None, float | None]:
        n_samples, n_features = X.shape
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features, int):
            k = min(n_features, self.max_features)
        else:
            k = n_features

        feature_indices = self.rng.choice(n_features, size=k, replace=False)
        
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        n_fraud = np.sum(y == 1)
        p_parent = n_fraud / n_samples
        parent_gini = 1.0 - (p_parent ** 2 + (1.0 - p_parent) ** 2)

        for feat in feature_indices:
            vals = X[:, feat]
            if len(vals) > 30:
                thresholds = np.percentile(vals, np.linspace(15, 85, 7))
                thresholds = np.unique(thresholds)
            else:
                thresholds = np.unique(vals)

            for thresh in thresholds:
                left_mask = vals <= thresh
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left

                if n_left == 0 or n_right == 0:
                    continue

                w1_left = np.sum(y[left_mask] == 1)
                w1_right = n_fraud - w1_left

                p_left = w1_left / n_left
                p_right = w1_right / n_right

                gini_left = 1.0 - (p_left ** 2 + (1.0 - p_left) ** 2)
                gini_right = 1.0 - (p_right ** 2 + (1.0 - p_right) ** 2)

                gain = parent_gini - ((n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> DecisionNode:
        n_samples = len(y)
        n_fraud = np.sum(y == 1)
        prob = float(n_fraud / n_samples) if n_samples > 0 else 0.0

        # Stop criteria
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or prob == 0.0
            or prob == 1.0
        ):
            return DecisionNode(prob=prob)

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return DecisionNode(prob=prob)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return DecisionNode(prob=prob)

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionNode(
            feature_idx=feat, threshold=thresh, left=left_child, right=right_child
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastDecisionTree":
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_row(self, node: DecisionNode, x: np.ndarray) -> float:
        if node.is_leaf:
            return node.prob
        if x[node.feature_idx] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(self.root, row) for row in X], dtype=np.float64)


class PureRandomForestClassifier:
    """
    Balanced Random Forest Classifier.
    Draws balanced bootstrap samples per tree (all minority cases + equal/3x majority samples).
    """
    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 7,
        min_samples_split: int = 8,
        max_features: str = "sqrt",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees: list[FastDecisionTree] = []
        self.rng = np.random.RandomState(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PureRandomForestClassifier":
        self.trees = []
        fraud_idx = np.where(y == 1)[0]
        legit_idx = np.where(y == 0)[0]

        n_fraud = len(fraud_idx)
        # 3x majority ratio for rich discrimination
        n_sample_legit = min(len(legit_idx), n_fraud * 4)

        for i in range(self.n_estimators):
            seed = self.rng.randint(0, 1_000_000)
            tree_rng = np.random.RandomState(seed)
            
            # Bootstrap minority (fraud) + subsample majority (legitimate)
            boot_fraud = tree_rng.choice(fraud_idx, size=n_fraud, replace=True)
            boot_legit = tree_rng.choice(legit_idx, size=n_sample_legit, replace=False)
            
            combined_idx = np.concatenate([boot_fraud, boot_legit])
            tree_rng.shuffle(combined_idx)

            X_b = X[combined_idx]
            y_b = y[combined_idx]

            tree = FastDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X_b, y_b)
            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, list):
            X = np.asarray(X, dtype=np.float64)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        tree_preds = np.array([tree.predict_proba(X) for tree in self.trees])
        p1 = np.mean(tree_preds, axis=0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> PureRandomForestClassifier:
    """Supervised fraud model training."""
    model = PureRandomForestClassifier(
        n_estimators=40,
        max_depth=7,
        min_samples_split=8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def fraud_probability(model: PureRandomForestClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
