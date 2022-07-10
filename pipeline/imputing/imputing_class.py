from pipeline.utils.singleton import Singleton
from pipeline.utils.logger import create_logger


@Singleton
class Imputing:

    logger = create_logger("imputing", "info")

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.sfa_methods = None

    def run(self, X):
        self.logger.info("run imputing")
        X = X.fillna(0)
        self.logger.info(f"after imputing {(X.isna().sum() > 0).sum()} #columns with missing values")
        return X