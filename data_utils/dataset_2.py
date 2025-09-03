import os
from torch.utils.data import Dataset
import torch
import numpy as np
import time
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
import torch.nn.functional as F
import transforms3d
# from .dataset_utils import rotate_pointcloud_and_hand_pose, euler2rot6d



def euler2rot6d(euler):
    rot = np.array(transforms3d.euler.euler2mat(
        *euler))
    ortho6d = rot[:, :2].T.ravel()
    return torch.from_numpy(ortho6d.astype(np.float32))


enable_inject_noise_V2 = False
pcl_noise_factor = 0
articulation_noise_factor = 0
trans_noise_factor = 0
rot_noise_factor = 0

if enable_inject_noise_V2:
    pcl_noise_factor = 0.002
    articulation_noise_factor = 0.002
    trans_noise_factor = 0.001
    rot_noise_factor = 0.01



class SynergyDatasetV2():
    def __init__(self, split, eigengrasp_head_cnt, max_dof, augment=False,use_whitened_eigengrasp=False, use_rot6d=False):
        # Initialize individual datasets
        self.allegro_dataset = SynergyDatasetAllegroHandV2(split, eigengrasp_head_cnt, max_dof,augment=augment,use_whitened_eigengrasp=use_whitened_eigengrasp, use_rot6d=use_rot6d)
        self.shadow_dataset = SynergyDatasetShadowHandV2(split, eigengrasp_head_cnt, max_dof, augment=augment,use_whitened_eigengrasp=use_whitened_eigengrasp,  use_rot6d=use_rot6d)
        # self.robotiq_3f_dataset = SynergyDatasetRobotiq3f(split, eigengrasp_head_cnt, max_dof)
        self.barrett_dataset = SynergyDatasetBarrettV2(split, eigengrasp_head_cnt, max_dof, augment=augment,use_whitened_eigengrasp=use_whitened_eigengrasp, use_rot6d=use_rot6d)
        # self.human_hand_dataset = SynergyDatasetHumanHand(split, eigengrasp_head_cnt, max_dof)
        # Combine datasets
        self.dataset = ConcatDataset([self.allegro_dataset,
                                      self.shadow_dataset,
                                      # self.robotiq_3f_dataset,
                                      self.barrett_dataset,
                                      # self.human_hand_dataset
                                      ])

        self.split = split  # Store split type for later use

        # Compute sampling weights for balanced sampling
        len_allegro = len(self.allegro_dataset)
        len_shadow = len(self.shadow_dataset)
        # len_robotiq_3f = len(self.robotiq_3f_dataset)
        len_barrett = len(self.barrett_dataset)
        # len_human_hand = len(self.human_hand_dataset)
        self.length = len_allegro + len_shadow + len_barrett
        print(f"total {self.length} data to train")
        if split == "train":
            weights = np.concatenate([
                np.full(len_allegro, 1 / len_allegro),  # Allegro samples
                np.full(len_shadow, 1 / len_shadow),  # Shadow samples
                # np.full(len_robotiq_3f, 1 / len_robotiq_3f),  # Robotiq 3f samples
                np.full(len_barrett, 1 / len_barrett) , # barrett samples
                # np.full(len_human_hand, 1 / len_human_hand)  # human hand
            ])

            self.sampler = WeightedRandomSampler(weights, num_samples=len(self.dataset), replacement=True)
        else:
            self.sampler = None  # No sampling for validation/testing

    def get_loader(self, batch_size, num_workers):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.sampler if self.split == "train" else None,  # Use sampler only for training
            shuffle=(self.split == "train" and self.sampler is None),  # Shuffle only if not using sampler
            num_workers=num_workers,
            pin_memory=True
        )

    def __len__(self):
        return self.length


class SynergyDatasetBarrettV2(Dataset):
    def __init__(self, split,  pcl_dir="/home/heng/DRO-Grasp/data/pointcloud_barrett", use_rot6d=True):
        print(f"loading barrett V2 {split} dataset")
        self.split = split
        # self.augment = augment
        self.pcl_dict = {}
        self.use_rot6d = use_rot6d

        files =  [f for f in os.listdir(pcl_dir) if f.endswith(".pt")]
        for file in files:
            pcl_data = torch.load(os.path.join(pcl_dir,file), weights_only=False)
            pcl = pcl_data["pcl"]
            pcl_key = file[:-3]
            self.pcl_dict[pcl_key] = torch.Tensor(pcl).float()
        start_time = time.time()
        path = f"/home/heng/DRO-Grasp/data/synergy/barrett/{split}_2.pt"
        # print(f"loading data from {path}")
        self.data = torch.load(path, weights_only=False)

        # eigengrasp_path = "/home/heng/synergy_grasp_sampler/data/dataset_barrett/eigengrasps_2.pt"
        # if use_whitened_eigengrasp:
        #     eigengrasp_path = eigengrasp_path.split(".")[0] + "_whitened.pt"
        # print(f"Loading eigengrasps from path {eigengrasp_path}")
        # self.max_dof = max_dof

        # raw_eigengrasp = torch.load(eigengrasp_path, weights_only=False)[:eigengrasp_head_cnt]
        # self.eigengrasp = F.pad(raw_eigengrasp, (0, max_dof-raw_eigengrasp.shape[1]), mode="constant", value=0).to(torch.float32)
        # print("eigengrasp shape", self.eigengrasp.shape)

        # pytorch_load_time = time.time()
        # print(f"spend {pytorch_load_time-start_time}s for loading barrett hand {split} data")
        # self.pcl_noise_factor = pcl_noise_factor

    def __len__(self):
        return len(self.data['pose'])

    # def __getitem__(self, idx):
    def __getitem__(self, idx):
        trans = self.data['pose'][idx][0:3].float()
        rot = self.data['pose'][idx][3:6].float()
        raw_articulation = self.data['pose'][idx][6:].float()
        # articulaetion = F.pad(raw_articulation, (0, self.max_dof-raw_articulation.shape[0]), mode="constant", value=0)
        grasp_code = str(self.data["grasp_code"][idx])
        # scale = self.data["scale"][idx]
        pcl = self.pcl_dict[grasp_code].float()
        # embodiment_id = self.data["embodiment_id"][idx].item()
        # translation_shift = self.data["translation_shift"][idx].float()
        # eigengrasps = self.eigengrasp.float()
        # if self.augment and self.split == "train":
            # print("running augmentation for shadow hand")
            # pcl, trans, rot = rotate_pointcloud_and_hand_pose(pcl, trans, rot)

        # if pcl_noise_factor != 0 and self.split == "train":
            # print("adding noise to pcl")
            # pcl = pcl + torch.randn_like(pcl) * pcl_noise_factor

        # if trans_noise_factor != 0 and self.split == "train":
            # print("adding noise to trans")

            # trans = trans + torch.randn_like(trans) * trans_noise_factor

        # if rot_noise_factor != 0 and self.split == "train":
            # print("adding noise to rot")

            # rot = rot + torch.randn_like(rot) * rot_noise_factor

        # if articulation_noise_factor != 0 and self.split == "train":
        #     # print("adding noise to articulation")

        #     articulation = articulation + torch.randn_like(articulation) * articulation_noise_factor

        if self.use_rot6d:
            rot = euler2rot6d(rot)


        return trans, rot, pcl, raw_articulation, grasp_code


class SynergyDatasetAllegroHandV2(Dataset):
    def __init__(self, split, pcl_dir="/home/heng/DRO-Grasp/data/pointcloud_allegro", use_rot6d=True):
        print(f"loading allegro hand V2 {split} dataset")
        self.pcl_dict = {}
        self.split = split
        self.use_rot6d = use_rot6d

        files = [f for f in os.listdir(pcl_dir) if f.endswith(".pt")]
        for file in files:
            pcl_data = torch.load(os.path.join(pcl_dir,file), weights_only=False)
            pcl = pcl_data["pcl"]
            pcl_key = file[:-3]
            self.pcl_dict[pcl_key] = torch.Tensor(pcl).float()
        self.data = torch.load(f"/home/heng/DRO-Grasp/data/synergy/allegro/{split}_2.pt", weights_only=False)

        # self.max_dof = max_dof

        # self.pcl_noise_factor = pcl_noise_factor

    def __len__(self):
        return len(self.data['pose'])

    def __getitem__(self, idx):
        trans = self.data['pose'][idx][0:3].float()
        rot = self.data['pose'][idx][3:6].float()
        raw_articulation = self.data['pose'][idx][6:].float()
        # articulation = F.pad(raw_articulation, (0, self.max_dof-raw_articulation.shape[0]), mode="constant", value=0)
        grasp_code = str(self.data["grasp_code"][idx])
        # scale = self.data["scale"][idx]
        pcl = self.pcl_dict[grasp_code].float()
        # embodiment_id = self.data["embodiment_id"][idx].item()
        # translation_shift = self.data["translation_shift"][idx].float()
        # eigengrasps = self.eigengrasp.float()
        # if self.augment and self.split == "train":
            # print("running augmentation for shadow hand")
            # pcl, trans, rot = rotate_pointcloud_and_hand_pose(pcl, trans, rot)

        # if pcl_noise_factor != 0 and self.split == "train":
            # print("adding noise to pcl")
            # pcl = pcl + torch.randn_like(pcl) * pcl_noise_factor

        # if trans_noise_factor != 0 and self.split == "train":
            # print("adding noise to trans")

            # trans = trans + torch.randn_like(trans) * trans_noise_factor

        # if rot_noise_factor != 0 and self.split == "train":
            # print("adding noise to rot")

            # rot = rot + torch.randn_like(rot) * rot_noise_factor

        # if articulation_noise_factor != 0 and self.split == "train":
        #     # print("adding noise to articulation")

        #     articulation = articulation + torch.randn_like(articulation) * articulation_noise_factor

        if self.use_rot6d:
            rot = euler2rot6d(rot)


        return trans, rot, pcl, raw_articulation, grasp_code



class SynergyDatasetShadowHandV2(Dataset):
    def __init__(self, split, pcl_dir="/home/heng/DRO-Grasp/data/pointcloud_shadow",  use_rot6d=True):
        print(f"loading shadow hand V2 {split} data......")
        self.pcl_dict = {}
        self.split = split
        # self.augment = augment
        self.use_rot6d = use_rot6d


        pcl_cache_path = os.path.join(pcl_dir, "pcl_dict_shadow.pt")
        if os.path.exists(pcl_cache_path):
            print(f"✅ Found cached point clouds → {pcl_cache_path}")
            self.pcl_dict = torch.load(pcl_cache_path)
        else:
            files = [f for f in os.listdir(pcl_dir) if f.endswith(".npy")]
            for file in files:
                pcl = torch.load(os.path.join(pcl_dir, file), weights_only=False)["pcl"]
                pcl_key = file[:-4]
                self.pcl_dict[pcl_key] = torch.Tensor(pcl)
                # print(self.pcl_dict[pcl_key].shape)
            torch.save(self.pcl_dict, pcl_cache_path)
            print(f"✅ Cached {len(self.pcl_dict)} point clouds → {pcl_cache_path}")

        # start_time = time.time()
        self.data = torch.load(f"/home/heng/DRO-Grasp/data/synergy/shadow/{split}_2.pt", weights_only=False)
        # eigengrasp_path = "/home/heng/synergy_grasp_sampler/data/dataset_shadow/eigengrasps_2.pt"
        # if use_whitened_eigengrasp:
            # eigengrasp_path = eigengrasp_path.split(".")[0] + "_whitened.pt"
        # print(f"Loading eigengrasps from path {eigengrasp_path}")
        # raw_eigengrasp = torch.load(eigengrasp_path)[:eigengrasp_head_cnt]
        # self.eigengrasp = F.pad(raw_eigengrasp, (0, max_dof-raw_eigengrasp.shape[1]), mode="constant", value=0).to(torch.float32)
        # self.max_dof = max_dof
        # print("eigengrasp shape", self.eigengrasp.shape)
        # pytorch_load_time = time.time()
        # print(f"spend {pytorch_load_time-start_time}s for loading {split} data")

        scale_tensor = self.data["scale"]  # shape (dataset_size, 1)
        scale_values = scale_tensor.view(-1).tolist()  # flatten to Python list

        # Convert to 2-decimal strings
        scale_str_list = [f"{s:.2f}" for s in scale_values]

        # Replace scale in self.data with string list
        self.data["scale"] = scale_str_list

        # self.pcl_noise_factor = pcl_noise_factor

    def __len__(self):
        return len(self.data['pose'])

    def __getitem__(self, idx):
        trans = self.data['pose'][idx][0:3]
        rot = self.data['pose'][idx][3:6]
        raw_articulation = self.data['pose'][idx][6:]
        # articulation = F.pad(raw_articulation, (0, self.max_dof-raw_articulation.shape[0]), mode="constant", value=0)

        grasp_code = str(self.data["grasp_code"][idx])
        scale = self.data["scale"][idx]

        pcl_key = grasp_code + "_" + str(scale)
        pcl = self.pcl_dict[pcl_key]
        # embodiment_id = self.data["embodiment_id"][idx].item()
        # if self.augment and self.split == "train":
            # print("running augmentation for shadow hand")
            # pcl, trans, rot = rotate_pointcloud_and_hand_pose(pcl, trans, rot)

        # if pcl_noise_factor != 0 and self.split == "train":
            # print("adding noise to pcl")
            # pcl = pcl + torch.randn_like(pcl) * pcl_noise_factor

        # if trans_noise_factor != 0 and self.split == "train":
            # print("adding noise to trans")

            # trans = trans + torch.randn_like(trans) * trans_noise_factor

        # if rot_noise_factor != 0 and self.split == "train":
            # print("adding noise to rot")

            # rot = rot + torch.randn_like(rot) * rot_noise_factor

        # if articulation_noise_factor != 0 and self.split == "train":
            # print("adding noise to articulation")

            # articulation = articulation + torch.randn_like(articulation) * articulation_noise_factor

        # translation_shift = self.data["translation_shift"][idx]

        # if self.use_rot6d:
            # rot = euler2rot6d(rot)

        return trans, rot, pcl, raw_articulation, pcl_key

if __name__ == "__main__":
    # dataset = SynergyDatasetShadowHand("test", 9, 22)
    # # torch_dataset = SynergyDataset("test", use_torch_load=True)
    # print("shadow dataset size: ", dataset.__len__())
    # trans, rot, pcl, articulation, embodiment_id, eigengrasp, translation_shift, object_code, scale= (dataset.__getitem__(19))
    # print("trans.shape: ", trans.shape)
    # print("trans.type: ", trans.type())
    # #
    # print("rot.shape: ", rot.shape)
    # print("rot.type: ", rot.type())
    #
    # print("pcl.shape: ", pcl.shape)
    # print("pcl.type: ", pcl.type())
    #
    # print("articulation.shape: ", articulation.shape)
    # print("articulation.type: ", articulation.type())
    #
    # print("embodiment_id.shape: ", embodiment_id.shape)
    # print("embodiment_id.type: ", embodiment_id.type())
    # #
    # # print("translation_shift.shape: ", translation_shift.shape)
    # # print("type translation_shit: ", translation_shift.type())
    # #
    # # print("eigengrasp.shape", eigengrasp.shape)
    # # print("eigengrasp[0].shape: ", eigengrasp[0].shape)
    # # print("eigengrasp[0].type: ", eigengrasp[0].type())
    # #
    # # print("object_code type", type(object_code))
    # #
    # # print("scale shape: ", scale.shape)
    # # print("scale type: ", type(scale))
    # dataset = SynergyDatasetAllegroHand("test", 9, 22)
    # print("allegro dataset size: ", dataset.__len__())
    # trans, rot, pcl, articulation, embodiment_id, eigengrasp, translation_shift, object_code, scale= (dataset.__getitem__(19))
    # print("trans.shape: ", trans.shape)
    # print("trans.type: ", trans.type())
    #
    # print("rot.shape: ",rot.shape)
    # print("rot.type: ",rot.type())
    #
    # print("pcl.shape: ", pcl.shape)
    # print("pcl.type: ", pcl.type())
    #
    # print("articulation.shape: ",articulation.shape)
    # print("articulation.type: ",articulation.type())
    #
    # print("embodiment_id.shape: ", embodiment_id.shape)
    # print("embodiment_id.type: ", embodiment_id.type())
    #
    # print("translation_shift.shape: ", translation_shift.shape)
    # print("type translation_shit: ", translation_shift.type())
    #
    # print("eigengrasp.shape", eigengrasp.shape)
    # print("eigengrasp[0].shape: ",eigengrasp[0].shape)
    # print("eigengrasp[0].type: ",eigengrasp[0].type())
    #
    # print("object_code type", type(object_code))
    #
    # # print("scale shape: ", scale.shape)
    # print("scale type: ", type(scale))
    dataset = SynergyDatasetShadowHandV2("train")
    print("shadow hand dataset size: ", dataset.__len__())
    trans, rot, pcl, raw_articulation, grasp_code = (
        dataset.__getitem__(19))
    print("trans.shape: ", trans.shape)
    print("trans.type: ", trans.type())

    print("rot.shape: ", rot.shape)
    print("rot.type: ", rot.type())

    print("pcl.shape: ", pcl.shape)
    print("pcl.type: ", pcl.type())

    # print("articulation.shape: ", articulation.shape)
    # print("articulation.type: ", articulation.type())


