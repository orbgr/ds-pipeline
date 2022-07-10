from dataclasses import dataclass
from typing import Any


@dataclass
class EVAL:
    main_metric: Any
    kfold: int


cfg_eval = EVAL(main_metric='roc_auc',
                kfold=5)