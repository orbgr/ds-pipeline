from dataclasses import dataclass
import os
from os.path import join as join_path
from datetime import datetime
from recipes.feature_selection import cfg_fs, FS
from recipes.algos import cfg_algos, Algos
from recipes.modeling import cfg_modeling, MDL
from recipes.imputing import cfg_impute, IMP
from recipes.evaluation import cfg_eval, EVAL

@dataclass
class CFG:
    root_path: str
    timestamp: type(datetime)
    cfg_fs: FS
    cfg_modeling: MDL
    cfg_impute: IMP
    cfg_eval: EVAL


root_path = join_path(os.getcwd(), "..")
timestamp = datetime.now()

cfg = CFG(root_path=root_path,
          timestamp=timestamp,
          cfg_fs=cfg_fs,
          cfg_eval=cfg_eval,
          cfg_modeling=cfg_modeling,
          cfg_impute=cfg_impute
          )