import numpy as np
import pandas as pd
from pipeline.utils.singleton import Singleton
from pipeline.utils.logger import create_logger
from tqdm import tqdm
from inspect import getmembers, isfunction
from pipeline.feature_selection import single_factor
from pipeline.feature_selection import multi_factor


@Singleton
class FeatureSelection:

    logger = create_logger("feature_selection", "info")

    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg
        self.sfa_table = pd.DataFrame()
        self.sfa_methods = {func[0]: func[1] for func in getmembers(single_factor, isfunction) }

        # self.mfa_methods = getmembers(multi_factor, isfunction)

    def run(self, X, y, methods=None, masks=None):
        """
        run SFA and then MFA

        :param pd.DataFrame X: dataset
        :param pd.Series y: target
        :param list methods: feature_selection types
        :param dict masks: feature_selection masks, aka filters with the mask name as key
        :return: best_features_list
        """
        if self.cfg is not None:
            methods = self.cfg.sfa_methods
            masks = self.cfg.masks

        for method in tqdm(methods):
            func = self.sfa_methods[method]
            res = func(X, y)
            self.sfa_table[method] = res

        best_features_list = X.columns.tolist()
        for name, mask in masks.items():
            n_columms_start = len(best_features_list)
            best_features_list = mask(self.sfa_table.loc[best_features_list]).index.tolist()
            self.logger.info(f"FeatureSelection of {name=} filterd  #{n_columms_start - len(best_features_list)} columns")

        return best_features_list

# df = pd.read_excel("../../data/dataset.xls", index_col=0, header=1)
# fs = FeatureSelection.instance()
# target_col = "default payment next month"
# methods = ["gini", "coverage", "uniq", "correlation"]
# masks = dict(gini=lambda df: df.loc[df["gini"] > 0.04],
#              coverage=lambda df: df.loc[df["coverage"] > 0.25])
#
# best_features_list = fs.run(df.drop(columns=[target_col]), df[target_col], methods, masks)
# print(best_features_list)