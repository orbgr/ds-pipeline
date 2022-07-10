from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from typing import Any
from catboost import CatBoostClassifier

@dataclass
class DefaultParams:
    name: str
    algo: Any
    params: dict

@dataclass
class Algos:
    pass


log_reg = DefaultParams(name="log_reg",
                        algo=LogisticRegression,
                        params=dict(random_state=0, C=1, class_weight='balanced', solver='lbfgs'))

catboost = DefaultParams(name="catboost",
                        algo=CatBoostClassifier,
                        params=dict(random_state=0, iterations=300, auto_class_weights='balanced'))

cfg_algos = Algos()
cfg_algos.catboost = catboost
cfg_algos.log_reg = log_reg
