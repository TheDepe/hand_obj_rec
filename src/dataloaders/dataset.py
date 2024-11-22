import os
import torchvision.io as io
import csv
from PIL import Image
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset

from utils.utils import list_dirs, plot_renderings, apply_mask
from utils.manus_cam_utils import *

class MirageDataPrep:
    def __init__(self, base_path, split=None, verbose=False):
        self.base_path = base_path
        self.verbose = verbose
    
    def build_index_list(self, target_depth=1):
        dirs_containing_renders = []
        if self.verbose:
            print(f"Searching {self.base_path} for scene folders.")
        for root, dirs, files in os.walk(self.base_path):
            relative_path = os.path.relpath(root, self.base_path)
            depth = relative_path.count(os.sep)
            if depth == target_depth:
                dirs_containing_renders.append(root)
        
        if len(dirs_containing_renders) == 0:
            raise ValueError("Couldn't Find any images")
        
        if self.verbose:
            print(f"Found {len(dirs_containing_renders)} scenes.")
        return dirs_containing_renders
        
    def test_train_split(self, dir_list, train_split=0.8, save_path=None):
            random.shuffle(dir_list)
            split_index = int(len(dir_list) * train_split)
            train_dirs = dir_list[:split_index]
            test_dirs = dir_list[split_index:]
            
            if save_path:
                with open(save_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['directory','set'])
                    # write training dirs with "train" label
                    for dir_path in train_dirs:
                        writer.writerow([dir_path, 'train'])
                    for dir_path in test_dirs:
                        writer.writerow([dir_path, 'test'])

                if self.verbose:
                    print(f"Wrote test_train_split to {save_path}")
            return train_dirs, test_dirs
    
    def get_test_train_split(self, save_path=None):
        directories = self.build_index_list(target_depth=2)
        train_dirs, test_dirs = self.test_train_split(directories,save_path=save_path)
        return train_dirs, test_dirs


class CameraData:
    
    def __init__(self, name, params, device='cuda'):
        self.name = name
        self.device = device
        
        self.params = params
        self.width = params['width']
        self.height = params['height']
        
        self.znear = 0.05
        self.zfar = 50
        
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.params['fy']))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32).to(device=self.device)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.zfar + self.znear) / (self.zfar - self.znear)
        self.proj_matrix[3, 2] = - (self.zfar * self.znear) / (self.zfar - self.znear)
        self.proj_matrix[2, 3] = 1
        
        intr, dist = get_intr(self.params)
        new_intr, roi = get_undistort_params(
            intr, dist, (self.width, self.height)
        )
        extr = get_extr(self.params)
        
        self.attr_dict = get_opengl_camera_attributes(
            new_intr,
            extr,
            self.width,
            self.height,
            resize_factor=1
        )

    @property
    def intrins(self):
        return torch.tensor(self.attr_dict['K']).float().to(device=self.device)
    
    @property
    def extrins(self):
        return torch.tensor(self.attr_dict['extr']).float().to(device=self.device)
        
    @property
    def view_matrix(self):
        return torch.tensor(self.attr_dict['world_view_transform']).float().to(device=self.device)
    
    @property
    def position(self):
        return torch.tensor(self.attr_dict['camera_center']).float().to(device=self.device)
    
    @property
    def view_proj_matrix(self):
        return torch.tensor(self.attr_dict['full_proj_transform']).float().to(device=self.device)
    
    
class CameraView:
    """
    Class to handle lazy-loading of rendering data.
    Arguments: 
        path: path to scene folder
        cam_data: CameraData
    
    """
    def __init__(self, path, cam_data, device='cuda'):
        self.base_path = path
        self.cam_name = self.base_path.split(os.sep)[-1]
        self.imread = Image.open
        self.cam_params = cam_data       
        self.device = device
        
    def __repr__(self):
        string = f"Camera:\t{self.cam_name}\nKeys:"
        property_names = [
        "cam_data",
        "scene_rgb", "hand_rgb", "obj_rgb",
        "scene_mask", "hand_mask", "hand_mask_with_obj_interaction", "obj_mask",
        "scene_depth", "hand_depth", "obj_depth",
        "scene_normals", "hand_normals", "obj_normals"
        ]
        for p in property_names:
            string += f"\t{p}\n"
        return string
    
    @property
    def cam_data(self):
        return CameraData(self.cam_name, self.cam_params, self.device)
    # RGBs
    @property
    def scene_rgb(self):
        img = torch.tensor(np.array(self.imread(os.path.join(self.base_path, 'full_rgb.webp')))).to(device=self.device)
        mask = self.scene_mask
        return apply_mask(img,mask).float().to(device=self.device)
    
    @property
    def hand_rgb(self):
        img = self.imread(os.path.join(self.base_path, 'hand_rgb.webp'))
        mask = self.hand_mask
        return apply_mask(img, mask).float().to(device=self.device)
    
    @property
    def obj_rgb(self):
        img = self.imread(os.path.join(self.base_path, 'obj_rgb.webp'))
        return torch.tensor(np.array(img),dtype=torch.uint8).float().to(device=self.device)
    # Masks
    @property
    def hand_mask(self):
        img = self.imread(os.path.join(self.base_path, 'hand_mask_no_inter.webp'))
        return self.get_alpha_channel(img).to(device=self.device)
    
    @property
    def hand_mask_with_obj_interaction(self):
        img = self.imread(os.path.join(self.base_path, 'hand_mask_obj_inter.webp'))
        return self.get_alpha_channel(img).to(device=self.device)
    
    @property
    def obj_mask(self):
        img = self.imread(os.path.join(self.base_path, 'object_mask.webp'))
        return self.get_alpha_channel(img).to(device=self.device)
    
    @property
    def scene_mask(self):
        hand_mask = self.hand_mask
        obj_mask = self.obj_mask
        scene_mask = torch.clip(hand_mask + obj_mask, 0, 1)
        return scene_mask.to(device=self.device)
    # depth
    @property
    def scene_depth(self):
        img = self.imread(os.path.join(self.base_path, 'full_depth_map0001.webp'))
        mask = self.scene_mask
        return apply_mask(img, mask).to(device=self.device)
    
    @property
    def hand_depth(self):
        img = self.imread(os.path.join(self.base_path, 'hand_depth_map0001.webp'))
        mask = self.hand_mask
        return apply_mask(img, mask).to(device=self.device)
    
    @property
    def obj_depth(self):
        img = self.imread(os.path.join(self.base_path, 'obj_depth_map0001.webp'))
        return torch.tensor(np.array(img),dtype=torch.uint8).float().to(device=self.device)
    
    @property
    def scene_normals(self):
        img = self.imread(os.path.join(self.base_path, 'full_normal_map.webp'))
        mask = self.scene_mask
        return apply_mask(img, mask)
    
    @property
    def hand_normals(self):
        img = self.imread(os.path.join(self.base_path, 'hand_normal_map.webp'))
        mask = self.hand_mask
        return apply_mask(img, mask)
    
    @property
    def obj_normals(self):
        img = self.imread(os.path.join(self.base_path, 'obj_normal_map.webp'))
        return torch.tensor(np.array(img),dtype=torch.uint8).float().to(device=self.device)
    
    def get_alpha_channel(self, img):
        R,G,B,A = img.split()
        A = np.array(A)
        return torch.Tensor(np.where(A > 0, 1, 0).astype(np.uint8)).to(device=self.device)
        
    
        
class MirageDataset(Dataset):
    def __init__(self, file_paths, manus_cam_data=None):
        super().__init__()
        self.directories = file_paths
        if not manus_cam_data:
            manus_cam_data = os.path.join(self.directories, "optim_params.txt")
        self.cam_data = read_params(manus_cam_data)       

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __len__(self):
        return len(self.directories)
    
    def __getitem__(self, index):
        rendering_root_path = self.directories[index] # eg graspxl_renderings/{obj_id}/{size}/{render_combo}/
        
        split_path = rendering_root_path.split(os.sep)
        object_id = split_path[-3]
        size = split_path[-2]
        render_combo = split_path[-1]
        texture = render_combo.split('_mano')[0]
        sequence = "mano"+render_combo.split('_mano')[-1]
        
        avail_cams = list_dirs(rendering_root_path) # camera names
        camera_views =  [CameraView(os.path.join(rendering_root_path,cam), self.cam_data[cam], self.device) for cam in avail_cams]
        input_view = random.choice(camera_views) # choose a random view as monocular input
        
        metadata = {
            "object_id": object_id,
            "object_scale": size,
            "texture": texture,
            "sequence": sequence
        }
        return (input_view, camera_views, metadata)
        
        
        
class MyDataModule(pl.LightningDataModule):
    def __init__(self, file_paths, manus_cam_data=None, batch_size=32, transform=None):
        super().__init__()
        self.file_paths = file_paths
        self.manus_cam_data = manus_cam_data
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        dataset = MirageDataset(self.file_paths, self.manus_cam_data)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_set, self.val_set = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)