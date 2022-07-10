from pipeline.utils.singleton import Singleton
from pipeline.utils.logger import create_logger
from scipy.special import expit


@Singleton
class Modeling:

    logger = create_logger("modeling", "info")

    def __init__(self, features_list, best_params, cfg=None, **kwargs):
        self.cfg = cfg
        self.features_list = features_list
        self.best_params = best_params
        self.calib_a, self.calib_b = None, None

        self.model = self.cfg.algo(**best_params)

    def fit(self, dataset):
        self.dataset = dataset
        self.X_train, self.y_train = dataset.X.loc[dataset.train_index, self.features_list], dataset.y.loc[dataset.train_index]
        self.X_test, self.y_test = dataset.X.loc[dataset.test_index, self.features_list], dataset.y.loc[dataset.test_index]

        category_cols = list(set(dataset.cols_cat) & set(self.features_list))

        _ = self.model.fit(self.X_train, self.y_train)

        # if self.caliberate:
        #     caliber = Calibration(type=None)
        #     self.calib_a, self.calib_b = caliber.fit([self.model], self.test_data, self.y_test)

        return self.model

    def predict(self, dataset):
        cat_features = list(set(self.features_list) & set(dataset.cols_cat))

        dataset.data["y_pred"] = self.model.predict_proba(dataset.X[self.features_list])[:, 1]

        # if self.caliberate:
        #     dataset.X["y_pred"] = dataset.X["y_pred"].apply(lambda x: self.sigmoid(x, self.calib_a, self.calib_b))

        return dataset.data["y_pred"]

    @staticmethod
    def _sigmoid(x, a, b):
        return expit(-(a * x + b))