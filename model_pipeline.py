from pipeline.utils.logger import create_logger
from pipeline.feature_selection.feature_selection_class import FeatureSelection
from pipeline.imputing.imputing_class import Imputing
from pipeline.modeling.modeling_class import Modeling
from pipeline.evaluation.eval_class import Evaluation
from recipes.cfg import cfg
import warnings
warnings.filterwarnings("ignore")


class ModelPipeline:
    logger = create_logger("pipeline", "info")

    def __init__(self, model_name, dataset, dataset_oot=None):
        self.cfg = cfg
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_oot = dataset_oot
        self.is_first_method_for_selected_features = True
        self.feature_list = self.dataset.X.columns.tolist()
        self.model = None

    def run(self, type):
        if type == "train":
            self.dataset.train_index, self.dataset.test_index = self.dataset.split_xy(self.dataset.X, self.dataset.y)
            # check if size enough

            # feature eng
            # if self.feature_eng:
            #     self.generate_features(self.cfg["feature_generation"])

            # imputing
            self.logger.info("Start imputing")
            imp = Imputing.instance(cfg=cfg.cfg_impute)
            self.dataset.set_data(imp.run(self.dataset.X))

            # feature selection
            self.logger.info("Start feature selection")
            fs = FeatureSelection.instance(cfg=cfg.cfg_fs)
            self.features_list = fs.run(self.dataset.X, self.dataset.y)

            # hyper parameter tuning
            # self.best_params = self.hyper_tune()
            self.best_params = cfg.cfg_modeling.params

            # fit
            model_class = Modeling.instance(cfg=cfg.cfg_modeling,
                                            best_params=self.best_params,
                                            features_list=self.features_list)
            self.model = model_class.fit(self.dataset)

            # predict
            y_pred = model_class.predict(self.dataset)

            # evaluate - generative power, calibre goodness, OOT, shap, pvalue, feature_interactions
            evaluation = Evaluation.instance(cfg=cfg.cfg_eval)
            evaluation.run(model=self.model,
                           dataset=self.dataset,
                           features_list=self.features_list)

            # self.monitor_res = self.monitor(self.cfg["monitor"])

            # self.save(type)
            
            return

            # decline reasons
            # different pop evaluation
            # make the code works in seperate parts
            # compare models

        elif type == "predict":
            pass
            # self.predict()
            # self.evaluate()
            # self.monitor()
            # self.save(type)

        elif type == "monitor":
            pass

        elif type is None:
            pass

        else:
            raise Exception("type problem")