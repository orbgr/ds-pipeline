from pipeline.utils.singleton import Singleton
from pipeline.utils.logger import create_logger
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

@Singleton
class Evaluation:

    logger = create_logger("evaluation", "info")
    metrics = dict(roc_auc=lambda y_true, y_pred: roc_auc_score(y_true, y_pred))

    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg

    def run(self, model, dataset, features_list):
        X_train = dataset.X.loc[dataset.train_index, features_list]
        # X_test = dataset.X.loc[dataset.test_index, features_list]
        y_train = dataset.y.loc[dataset.train_index]
        y_test = dataset.y.loc[dataset.test_index]

        self.logger.info("eval run Cross Validation")
        cv_mean, cv_std = self.cross_validation(model=model,
                                                X=X_train,
                                                y=y_train)
        self.logger.info(f"{cv_mean=} +- {cv_std * 2}")

        self.logger.info("eval run IN SAMPLE")
        in_sample = self.metrics[self.cfg.main_metric](y_train, dataset.data.loc[dataset.train_index, "y_pred"])
        self.logger.info(f"{in_sample=}")

        self.logger.info("eval run OOS SAMPLE")
        oos_sample = self.metrics[self.cfg.main_metric](y_test, dataset.data.loc[dataset.test_index, "y_pred"])
        self.logger.info(f"{oos_sample=}")
        # self.logger.info("eval run OOT SAMPLE")

    def cross_validation(self, model, X, y):
        scores = cross_val_score(model, X, y,
                                 cv=self.cfg.kfold,
                                 scoring=self.cfg.main_metric)

        return scores.mean(), scores.std()

    def shap_eval(self):
        pass