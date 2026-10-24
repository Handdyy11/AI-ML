"""XGBoost classifier adapter for the project's string-valued class labels."""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src import config


class StringLabelXGBClassifier(ClassifierMixin, BaseEstimator):
    """Encode labels for XGBoost while returning original labels from predict."""

    def __init__(
        self,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=config.SEED,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.objective = objective
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.random_state = random_state

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value
        if hasattr(self, "estimator_"):
            self.estimator_.set_params(n_jobs=value)

    def fit(self, X, y, sample_weight=None):
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        self.estimator_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            objective=self.objective,
            eval_metric=self.eval_metric,
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.estimator_.fit(
            X,
            self.label_encoder_.transform(y),
            sample_weight=sample_weight,
        )
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        encoded_predictions = self.estimator_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(encoded_predictions)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_
