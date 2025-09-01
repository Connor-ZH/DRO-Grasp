import os
import sys
import time
import random
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


def replace_euler_with_rot6d(poses: torch.Tensor, rot_6ds: torch.Tensor):
    translations = poses[:, :3]      # first 3 → translation
    articulations = poses[:, 6:]     # skip translation (3) + euler (3)
    new_poses = torch.cat([translations, rot_6ds, articulations], dim=1)
    return new_poses



class PretrainDataset(Dataset):
    def __init__(self, robot_names: list = None, dataset_type = ""):
        self.robot_names = robot_names if robot_names is not None \
            else ['barrett', 'allegro', 'shadowhand']


        assert dataset_type in ["DRO", "SYNERGY"]
        if dataset_type == "SYNERGY":
            self.robot_names = ['barrett', 'allegro', 'shadow']
        self.dataset_type = dataset_type
        self.dataset_len = 0
        self.robot_len = {}
        self.hands = {}
        self.dofs = []
        self.dataset = {}
        for robot_name in self.robot_names:


            self.hands[robot_name] = create_hand_model(robot_name, torch.device('cpu'), dataset_type=dataset_type)
            self.dofs.append(len(self.hands[robot_name].pk_chain.get_joint_parameter_names()))
            self.dataset[robot_name] = []
            if dataset_type == "DRO":
                dataset_path = os.path.join(ROOT_DIR, f'data/MultiDex_filtered/{robot_name}/{robot_name}.pt')
                dataset = torch.load(dataset_path)
                pose_key = "metadata"
                metadata = dataset[pose_key]
            else:
                dataset_path = os.path.join(ROOT_DIR, f'data/synergy/{robot_name}/train_2.pt')
                dataset = torch.load(dataset_path)
                pose = dataset["pose"]
                rot_6ds = dataset["rot_6ds"]
                metadata = replace_euler_with_rot6d(pose, rot_6ds)

            self.dataset[robot_name].extend(metadata)
            self.dataset_len += len(metadata)
            self.robot_len[robot_name] = len(metadata)



    def __getitem__(self, index):
        robot_name = random.choices(self.robot_names, weights=self.dofs, k=1)[0]

        hand = self.hands[robot_name]
        dataset = self.dataset[robot_name]
        if self.dataset_type == "DRO":
            target_q, _, _ = random.choice(dataset)
        else:
            target_q = random.choice(dataset).float()

        # print("hand: ", robot_name)
        # print("target_q: ", target_q.shape)
        # raise "stop"
        robot_pc_1 = hand.get_transformed_links_pc(target_q)[:, :3]
        initial_q = hand.get_initial_q(target_q)
        robot_pc_2 = hand.get_transformed_links_pc(initial_q)[:, :3]

        return {
            'robot_pc_1': robot_pc_1,
            'robot_pc_2': robot_pc_2,
        }

    def __len__(self):
        return self.dataset_len


def create_dataloader(cfg):
    dataset_type = cfg.dataset_type
    print(f"using {dataset_type} to pretrain model")

    dataset = PretrainDataset(cfg.robot_names, dataset_type)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=True
    )
    return dataloader
