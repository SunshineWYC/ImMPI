import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


class MVSDataset(Dataset):
    def __init__(self, dataset_filepath, list_filepath, neighbor_view_num=4):
        super(MVSDataset, self).__init__()

        self.dataset_filepath = dataset_filepath
        self.list_filepath = list_filepath
        self.neighbor_view_num = neighbor_view_num

        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.list_filepath, "r") as f:
            scene_ids = f.readlines()
            scene_ids = [scene_id.rstrip() for scene_id in scene_ids]
        if self.neighbor_view_num == 4:
            for scene_id in scene_ids:  # fixed sample type, 6 metas per scene
                metas.append([scene_id, "006", "005", "007", "001", "011"])
                metas.append([scene_id, "007", "006", "008", "002", "012"])
                metas.append([scene_id, "008", "007", "009", "003", "013"])
                metas.append([scene_id, "011", "010", "012", "006", "016"])
                metas.append([scene_id, "012", "011", "013", "007", "017"])
                metas.append([scene_id, "013", "012", "014", "008", "018"])
        if self.neighbor_view_num == 19:
            for scene_id in scene_ids:  # fixed sample type, 2 metas per scene
                metas.append([scene_id, "007", "002", "006", "008", "012", "000", "001", "003", "004", "005", "009", "010", "011", "013", "014", "015", "016", "017", "018", "019"])
                metas.append([scene_id, "012", "007", "011", "013", "017", "000", "001", "002", "003", "004", "005", "006", "008", "009", "010", "014", "015", "016", "018", "019"])
        return metas

    def read_image(self, filename):
        image = Image.open(filename)
        # normalize 0-255 to 0-1
        image = np.array(image, dtype=np.float32) / 255.
        return image

    def read_depth(self, filename):
        depth = np.array(Image.open(filename)).astype(np.float32)
        return depth / 64.0

    def read_camera_parameters(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsic = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape(4, 4)
        intrinsic = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape(3, 3)
        depth_min, depth_max = [float(item) for item in lines[11].split()]
        return intrinsic, extrinsic, depth_min, depth_max

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        scene_id, sample_id = self.metas[index][0], self.metas[index][1]
        neighbor_view_ids = self.metas[index][2:]
        image_ref = self.read_image(os.path.join(self.dataset_filepath, scene_id, "Images/{}.png".format(sample_id))).transpose([2, 0, 1])  # CHW
        depth_ref = self.read_depth(os.path.join(self.dataset_filepath, scene_id, "Depths/{}.png".format(sample_id)))
        K_ref, E_ref, depth_min_ref, depth_max_ref = self.read_camera_parameters(os.path.join(self.dataset_filepath, scene_id, "Cams/{}.txt".format(sample_id)))

        images_tgt, Ks_tgt, Ts_tgt_ref = [], [], []
        for neighbor_view_id in neighbor_view_ids:
            images_tgt.append(self.read_image(os.path.join(self.dataset_filepath, scene_id, "Images/{}.png".format(neighbor_view_id))))
            K_tgt, E_tgt, depth_min_tgt, depth_max_tgt = self.read_camera_parameters(os.path.join(self.dataset_filepath, scene_id, "Cams/{}.txt".format(neighbor_view_id)))
            Ks_tgt.append(K_tgt)
            Ts_tgt_ref.append(np.matmul(E_tgt, np.linalg.inv(E_ref)))
        images_tgt = np.stack(images_tgt).transpose([0, 3, 1, 2])
        Ks_tgt = np.stack(Ks_tgt)
        Ts_tgt_ref = np.stack(Ts_tgt_ref)

        return {
            "scene_id": scene_id,  # str
            "sample_id": sample_id,  # str
            "image_ref": image_ref,  # [3, H, W], np.array
            "K_ref": K_ref,  # [3, 3], np.array
            "depth_min_ref": depth_min_ref,  # float
            "depth_max_ref": depth_max_ref,  # float
            "depth_ref": depth_ref,  # [H, W], np.array
            "images_tgt": images_tgt,  # [nNeighbor, 3, H, W], np.array
            "Ks_tgt": Ks_tgt,  # [nNeighbor, 3, 3], np.array
            "Ts_tgt_ref": Ts_tgt_ref,  # [nNeighbor, 4, 4], np.array
        }


class SceneData():
    def __init__(self, dataset_filepath, scene_id, sample_id, neighbor_view_num=19):
        super(SceneData, self).__init__()
        self.dataset_filepath = dataset_filepath
        self.scene_id = scene_id
        self.sample_id = sample_id
        self.neighbor_view_num = neighbor_view_num

        # build neighbor view_id list
        assert self.sample_id == "007" or sample_id == "012"
        if self.neighbor_view_num == 19:
            if self.sample_id == "007":
                self.neighbor_view_ids = ["002", "006", "008", "012", "000", "001", "003", "004", "005", "009", "010", "011", "013", "014", "015", "016", "017", "018", "019"]
            if self.sample_id == "012":
                self.neighbor_view_ids = ["007", "011", "013", "017", "000", "001", "002", "003", "004", "005", "006", "008", "009", "010", "014", "015", "016", "018", "019"]
        if self.neighbor_view_num == 4:
            if self.sample_id == "007":
                self.neighbor_view_ids = ["002", "006", "008", "012"]
            if self.sample_id == "012":
                self.neighbor_view_ids = ["007", "011", "013", "017"]

        self.scene_data = self.loadSceneData()

    def read_image(self, filename):
        image = Image.open(filename)
        # normalize 0-255 to 0-1
        image = np.array(image, dtype=np.float32) / 255.
        return image

    def read_depth(self, filename):
        depth = np.array(Image.open(filename)).astype(np.float32)
        return depth / 64.0

    def read_camera_parameters(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsic = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape(4, 4)
        intrinsic = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape(3, 3)
        depth_min, depth_max = [float(item) for item in lines[11].split()]
        return intrinsic, extrinsic, depth_min, depth_max

    def loadSceneData(self):
        image_ref = self.read_image(os.path.join(self.dataset_filepath, self.scene_id, "Images/{}.png".format(self.sample_id))).transpose([2, 0, 1])  # CHW
        depth_ref = self.read_depth(os.path.join(self.dataset_filepath, self.scene_id, "Depths/{}.png".format(self.sample_id)))
        K_ref, E_ref, depth_min_ref, depth_max_ref = self.read_camera_parameters(os.path.join(self.dataset_filepath, self.scene_id, "Cams/{}.txt".format(self.sample_id)))

        images_tgt, Ks_tgt, Ts_tgt_ref = [], [], []
        for neighbor_view_id in self.neighbor_view_ids:
            images_tgt.append(self.read_image(os.path.join(self.dataset_filepath, self.scene_id, "Images/{}.png".format(neighbor_view_id))))
            K_tgt, E_tgt, depth_min_tgt, depth_max_tgt = self.read_camera_parameters(os.path.join(self.dataset_filepath, self.scene_id, "Cams/{}.txt".format(neighbor_view_id)))
            Ks_tgt.append(K_tgt)
            Ts_tgt_ref.append(np.matmul(E_tgt, np.linalg.inv(E_ref)))
        images_tgt = np.stack(images_tgt).transpose([0, 3, 1, 2])
        Ks_tgt = np.stack(Ks_tgt)
        Ts_tgt_ref = np.stack(Ts_tgt_ref)

        # convert to torch.Tensor, add batch dim
        image_ref = torch.from_numpy(image_ref).unsqueeze(0)
        K_ref = torch.from_numpy(K_ref).unsqueeze(0)
        depth_min_ref = torch.tensor(depth_min_ref, dtype=torch.float32).unsqueeze(0)
        depth_max_ref = torch.tensor(depth_max_ref, dtype=torch.float32).unsqueeze(0)
        depth_ref = torch.from_numpy(depth_ref).unsqueeze(0)
        images_tgt = torch.from_numpy(images_tgt).unsqueeze(0)
        Ks_tgt = torch.from_numpy(Ks_tgt).unsqueeze(0)
        Ts_tgt_ref = torch.from_numpy(Ts_tgt_ref).unsqueeze(0)

        return {
            "scene_id": self.scene_id,  # str
            "sample_id": self.sample_id,  # str
            "image_ref": image_ref,  # [1, 3, H, W], torch.Tensor
            "K_ref": K_ref,  # [1, 3, 3], torch.Tensor
            "depth_min_ref": depth_min_ref,  # [1,], torch.Tensor
            "depth_max_ref": depth_max_ref,  # [1,], torch.Tensor
            "depth_ref": depth_ref,  # [1, H, W], torch.Tensor
            "images_tgt": images_tgt,  # [1, nNeighbor, 3, H, W], torch.Tensor
            "Ks_tgt": Ks_tgt,  # [1, nNeighbor, 3, 3], torch.Tensor
            "Ts_tgt_ref": Ts_tgt_ref,  # [1, nNeighbor, 4, 4], torch.Tensor
        }


if __name__ == '__main__':
    print("WHUViewSyn dataset")
    dataset_dirpath = r"D:\Datasets\WHU\WHUViewSyn\small_scale"
    list_filepath = "datalist/whuViewSyn/train.txt"
    neighbor_view_num = 4
    dataset = MVSDataset(dataset_dirpath, list_filepath, neighbor_view_num=neighbor_view_num)

    sample = dataset[320]
    print(sample["scene_id"], sample["sample_id"])
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(key, value.shape)
        else:
            print(key, value)
