import os
import csv
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from sklearn.model_selection import train_test_split
import random
from collections import Counter

class HybridTrainPipeline_140k(Pipeline):
    def __init__(
        self,
        data_list,  # List of (img_path, label) tuples
        root_dir,
        batch_size,
        num_threads,
        device_id=0,
        local_rank=0,
        world_size=1,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=-1)
        self.data_list = data_list
        self.root_dir = root_dir
        self.input = fn.readers.file(
            files=[os.path.join(root_dir, img_path) for img_path, _ in data_list],
            labels=[int(label in ['real', 'Real', 1]) for _, label in data_list],
            shard_id=local_rank,
            num_shards=world_size,
            random_shuffle=True,
        )
        self.decode = fn.decoders.image(device="mixed", output_type=types.RGB)
        self.res = fn.random_resized_crop(
            device="gpu", size=256, random_area=[0.8, 1.2], random_aspect_ratio=[0.9, 1.1]
        )
        self.cmnp = fn.crop_mirror_normalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            image_type=types.RGB,
            mean=[0.5207 * 255, 0.4258 * 255, 0.3806 * 255],
            std=[0.2490 * 255, 0.2239 * 255, 0.2212 * 255],
        )
        self.coin = fn.random.coin_flip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        images, labels = self.input
        images = self.decode(images)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, labels.gpu()]

class HybridValPipeline_140k(Pipeline):
    def __init__(
        self,
        data_list,  # List of (img_path, label) tuples
        root_dir,
        batch_size,
        num_threads,
        device_id=0,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=-1)
        self.data_list = data_list
        self.root_dir = root_dir
        self.input = fn.readers.file(
            files=[os.path.join(root_dir, img_path) for img_path, _ in data_list],
            labels=[int(label in ['real', 'Real', 1]) for _, label in data_list],
            random_shuffle=False,
        )
        self.decode = fn.decoders.image(device="mixed", output_type=types.RGB)
        self.res = fn.resize(
            device="gpu", resize_x=256, resize_y=256, interp_type=types.INTERP_TRIANGULAR
        )
        self.cmnp = fn.crop_mirror_normalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            image_type=types.RGB,
            mean=[0.5207 * 255, 0.4258 * 255, 0.3806 * 255],
            std=[0.2490 * 255, 0.2239 * 255, 0.2212 * 255],
        )

    def define_graph(self):
        images, labels = self.input
        images = self.decode(images)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, labels.gpu()]

class Dataset_140k_dali:
    def __init__(
        self,
        dataset_dir,
        train_batch_size,
        eval_batch_size,
        num_threads=4,
        device_id=0,
        local_rank=0,
        world_size=1,
    ):
        # Read CSV files
        def read_csv_file(csv_file, path_column='path', label_column='label'):
            data = []
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append((row[path_column], row[label_column]))
            return data

        train_csv = os.path.join(dataset_dir, 'train.csv')
        valid_csv = os.path.join(dataset_dir, 'valid.csv')
        test_csv = os.path.join(dataset_dir, 'test.csv')
        root_dir = os.path.join(dataset_dir, 'real_vs_fake', 'real-vs-fake')

        train_data = read_csv_file(train_csv)
        val_data = read_csv_file(valid_csv)
        test_data = read_csv_file(test_csv)

        # Debug: Print data statistics
        print("140k dataset statistics:")
        print(f"Total train dataset size: {len(train_data)}")
        print(f"Train label distribution: {Counter([x[1] for x in train_data])}")
        print(f"Total validation dataset size: {len(val_data)}")
        print(f"Total test dataset size: {len(test_data)}")

        # Create pipelines
        pipeline_train = HybridTrainPipeline_140k(
            data_list=train_data,
            root_dir=root_dir,
            batch_size=train_batch_size,
            num_threads=num_threads,
            device_id=device_id,
            local_rank=local_rank,
            world_size=world_size,
        )
        pipeline_train.build()
        self.loader_train = DALIGenericIterator(
            pipeline_train,
            ["data", "label"],
            size=len(train_data) // world_size,
            auto_reset=True,
        )

        pipeline_val = HybridValPipeline_140k(
            data_list=val_data,
            root_dir=root_dir,
            batch_size=eval_batch_size,
            num_threads=num_threads,
            device_id=device_id,
        )
        pipeline_val.build()
        self.loader_val = DALIGenericIterator(
            pipeline_val,
            ["data", "label"],
            size=len(val_data),
            auto_reset=True,
        )

        pipeline_test = HybridValPipeline_140k(
            data_list=test_data,
            root_dir=root_dir,
            batch_size=eval_batch_size,
            num_threads=num_threads,
            device_id=device_id,
        )
        pipeline_test.build()
        self.loader_test = DALIGenericIterator(
            pipeline_test,
            ["data", "label"],
            size=len(test_data),
            auto_reset=True,
        )
