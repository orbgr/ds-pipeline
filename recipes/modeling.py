from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from typing import Any
from catboost import CatBoostClassifier
from recipes.algos import cfg_algos

@dataclass
class MDL:
    algo: Any
    params: dict
    calib: bool = False


algo_class = cfg_algos.log_reg

cfg_modeling = MDL(algo=algo_class.algo,
                   params=algo_class.params)

