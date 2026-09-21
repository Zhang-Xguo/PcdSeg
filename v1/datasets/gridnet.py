import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import TRANSFORMS, index_operator


@TRANSFORMS.register_module()
class ClassAwareSphereCrop:
    """Crop a local neighborhood around a sampled rare-class point."""

    def __init__(
        self,
        point_max=150000,
        class_ids=(0, 1, 2, 4),
        class_weights=(0.25, 0.25, 0.4, 0.1),
        application_ratio=0.85,
    ):
        self.point_max = point_max
        self.class_ids = np.asarray(class_ids, dtype=np.int64)
        self.class_weights = np.asarray(class_weights, dtype=np.float64)
        self.application_ratio = application_ratio
        if len(self.class_ids) != len(self.class_weights):
            raise ValueError("class_ids and class_weights must have equal length")

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        if len(coord) <= self.point_max:
            return data_dict
        segment = np.asarray(data_dict["segment"]).reshape(-1)
        available = np.asarray([np.any(segment == c) for c in self.class_ids])
        if np.random.rand() < self.application_ratio and np.any(available):
            weights = self.class_weights * available
            target_class = np.random.choice(self.class_ids, p=weights / weights.sum())
            candidates = np.flatnonzero(segment == target_class)
            center_index = np.random.choice(candidates)
        else:
            center_index = np.random.randint(len(coord))
        distance = np.sum(np.square(coord - coord[center_index]), axis=1)
        crop = np.argpartition(distance, self.point_max - 1)[: self.point_max]
        return index_operator(data_dict, crop)


@DATASETS.register_module()
class GridNetDataset(DefaultDataset):
    """GridNet tiles with RGB luminance exposed as lidar-like strength."""

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        if "color" not in data_dict:
            raise KeyError("GridNet tile does not contain color.npy")
        rgb = data_dict["color"].astype(np.float32)
        data_dict["strength"] = (
            rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32) / 255.0
        )[:, None]
        return data_dict
