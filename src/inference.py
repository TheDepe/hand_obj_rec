import os
import matplotlib.pyplot as plt
from utils.utils import plot_renderings, dump_points
from dataloaders.dataset import MirageDataPrep, MirageDataset
from dataclasses import dataclass
from models.model import GaussianModel
from models.gaussian_renderer import GaussianRenderer
import torch
from torchvision.transforms import ToPILImage
from config.options import Options

from torch.utils.data import DataLoader #, random_split, Dataset

def main():
    base_path = '/home/perrettde/Documents/thesis/DATA/mirage_renders/new_renderings/'
    manus_cam_data = '/home/perrettde/Documents/thesis/DATA/Manus_Data/optim_params.txt'
    
    output_folder = '/home/perrettde/Documents/thesis/MIRAGE/hand_object_model/outputs'
    
    prep_pipeline = MirageDataPrep(base_path=base_path, verbose=True)
    train_dirs, test_dirs = prep_pipeline.get_test_train_split("test_train_split.csv")

    dataset = MirageDataset(file_paths=train_dirs, manus_cam_data=manus_cam_data)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Iterate through batches
    for inputs, targets, metadata in dataloader:
        print(f"Inputs: {inputs.shape}, Targets: {targets.shape}")
    
    #batch = dataset[0:3]
    #inputs, _, metadata = batch
    #model = GaussianModel()
    
    #input_image = inputs.scene_rgb.unsqueeze(0)
    #gaussians = model(input_image)
    
    #print(f"Number of gaussians: {gaussians.shape[1]}")
    
    #options = Options
    #renderer = GaussianRenderer(options)
    
    
    #view = inputs.cam_data.view_matrix.unsqueeze(0).unsqueeze(0)
    #view_proj = inputs.cam_data.view_proj_matrix.unsqueeze(0).unsqueeze(0)
    #pos = inputs.cam_data.position.unsqueeze(0).unsqueeze(0)
    
    #print(f"Camera Name: {inputs.cam_data.name}")
    #print(f"Camera View: {view}")
    #print(f"Camera position: {pos}")
    # gaussians: A tensor of predicted gaussians. Shape (B,N,D)
    # cam_view: A tensor of camera extrinsics. Shape (B,V, 4, 4)
    # cam_voew_proj: A tensor of combined cam_view and projection matrix. Shape (B, V, 4, 4)
    # cam_pos: A tensor of camera positions in world coordinates. Shape (B, V, 3)
    #results = renderer.render(
    #    gaussians=gaussians,
    #    cam_view=view,
    #    cam_view_proj=view_proj,
    #    cam_pos=pos,
    #    bg_color=torch.randn(3, dtype=torch.float32, device=gaussians.device),
    #    scale_modifier=0.1)
    

    
    #rgb = results['image'].squeeze(0).squeeze(0)
    #converter = ToPILImage()
    #image = converter(rgb)
    #image.save(os.path.join(output_folder,"test_render.png"))
    #dump_points(gaussians[0][..., :3], os.path.join(output_folder, "points.ply"))

if __name__ == "__main__":
    main()
    
