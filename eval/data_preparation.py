from data.kitti_generator import Dataset

DatasetEval = Dataset(task="validation",
                      validation=True,
                      evaluation=False)
