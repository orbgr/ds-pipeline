import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.utils.logger import create_logger


class ModelDataset:
    logger = create_logger("model_dataset", "info")

    def __init__(self,  col_target, col_index,
                 col_load_date=None, path=None,
                 type=None, data=None,
                 cols_cat=None, cols_metadata=None,
                 cols_date=None, cols_list=None,
                 cols_vector=None, **kwargs):
        """
        :param path: the query / file path
        :param type: path type: csv/query/dir
        :param col_target: target columns
        :param col_index: index column [str] todo: list
        :param cols_cat: feature_eng_cols
        :param cols_metadata: not features
        :param cols_date: feature_eng_cols
        :param cols_list: feature_eng_cols
        :param cols_vector: feature_eng_cols
        :param kwargs:
        """
        if (data is None) and (path is None):
            raise Exception("mush provide data or path")

        self.cols_metadata = cols_metadata
        self.cols_metadata = [] if cols_metadata is None else cols_metadata
        self.cols_cat = [] if cols_cat is None else cols_cat

        if data is not None:
            self.data, self.data_raw = data, data
        else:
            self.data, self.data_raw = self.get_data(path, type), data

        self.col_target = col_target
        self.col_load_date = col_load_date
        self.cols_date = cols_date
        self.col_index = col_index
        self.cols_list = cols_list
        self.cols_vector = cols_vector
        self.data.set_index(self.col_index, inplace=True)

        self.X, self.y, self.metadata = self.split_ds()

        self.data_force_numeric()

        self.cat_data_to_str()

        self.final_bins = None  # y_pred_rank bins
        # self.dataset_full, self.dataset_train, self.dataset_test = None, None, None  # will update in split

        self.qa_checks()

        # self.model_final_cols = None

    def cat_data_to_str(self):
        self.X[self.cols_cat] = self.X[self.cols_cat].astype(str)
        self.set_data(self.X)

    def data_force_numeric(self):
        for col in self.X.columns:
            self.X[col] = self.X[col].astype(float, errors='ignore')

        self.set_data(self.X)

    def qa_checks(self):
        # target missing
        data = pd.concat([self.X, self.y], axis=1)
        self.logger.info(f"data shape before cleansing {data.shape=}")

        data = data.dropna(subset=[self.col_target])
        self.logger.info(f"data shape after target missing: {data.shape=}")

        data = data.loc[~data.index.duplicated(keep='first'), ~data.columns.duplicated(keep='first')]
        data = data.drop(columns=data.filter(regex="Unnamed").columns)
        self.logger.info(f"data shape after duplicates: {data.shape=}")

        # missing all rows
        data = data.dropna(how='all')
        self.logger.info(f"data shape after all null records: {data.shape=}")

        y = data[self.col_target]
        X = data.drop(columns=[self.col_target])
        self.set_data(X, y)

    def set_data(self, X, y=None):
        self.X = X

        if y is not None:
            self.y = y

        self.data = pd.concat([self.X, self.y], axis=1)

    def get_data(self, path, type):
        if type == "query":
            pass
        elif type == "dir":
            pass
        elif type == "csv":
            data = pd.read_csv(path)
        elif type == "excel":
            data = pd.read_excel(path, header=1)
        else:
            raise Exception("problem with type")

        return data

    def split_ds(self):
        metadata = self.data[self.cols_metadata]
        X = self.data.drop(columns=self.cols_metadata + [self.col_target])
        y = self.data[self.col_target]

        return X, y, metadata

    @staticmethod
    def split_xy(X, y, test_size=0.3, stratify=None):
        stratify = y if stratify is None else stratify
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=42,
                                                            stratify=stratify)

        return X_train.index, X_test.index