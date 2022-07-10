import datetime
from os.path import join as join_path
from model_pipeline import ModelPipeline
from model_dataset import ModelDataset
import pickle

cols_cat = []
cols_metadata = []
cols_date = []
cols_vector = []
cols_list = []
col_index = "ID"
col_target = "default payment next month"


def modeling_flow():
    dataset = ModelDataset(path="./data/dataset.xls", type="excel",
                           col_target=col_target, col_index=col_index, col_load_date=None,
                           cols_cat=cols_cat, cols_metadata=cols_metadata, cols_date=cols_date,
                           cols_vector=cols_vector, cols_list=cols_list)

    model_pipeline = ModelPipeline(model_name="DSG", dataset=dataset, dataset_oot=None)
    model_pipeline.run(type="train")

    # models_pipeline_path = join_path(model_pipeline.model_name,
    #                                  datetime.datetime.today().strftime('%Y-%m-%d'),
    #                                  "models", "pipeline.pickle")
    # with open(models_pipeline_path, "wb") as file:
    #     pickle.dump(model_pipeline, file)


if __name__ == '__main__':    
    modeling_flow()
