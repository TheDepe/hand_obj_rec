import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from omegaconf import OmegaConf
from utils.gaussian_utils import strip_symmetric, build_scaling_rotation, build_symmetric

class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super(SimpleUNet, self).__init__()
        # Simple convolution to mimic the input and output behavior
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Group normalization to keep it consistent with the UNet structure
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        # A basic activation function
        self.activation = nn.SiLU()

    def forward(self, x, timestep=None):
        # Perform a simple forward pass
        x = self.conv(x)  # Convolution operation
        x = self.norm(x)  # Normalization
        x = self.activation(x)  # Activation function
        return x

class GaussianModel(torch.nn.Module):
    """
    a class representing a gaussian model.

    args:
        opts (object): an object containing options for the gaussian model.
        points (numpy.ndarray, optional): an array of points. if not provided, random points will be generated.
        points_colors (numpy.ndarray, optional): an array of colors for the points. if not provided, random colors will be generated.
    """
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_gaussians = 150
        
        self.unet = SimpleUNet(in_channels=4, out_channels=14).to(device)
        self.conv1 = nn.Conv2d(in_channels=14, out_channels=14, kernel_size=25).to(device)
        self.conv2 = nn.Conv2d(in_channels=14, out_channels=14, kernel_size=25).to(device)
        self.conv3 = nn.Conv2d(in_channels=14, out_channels=14, kernel_size=25).to(device)
        
        self.compression_layer = nn.Linear(782784,self.num_gaussians).to(device)
        self.splat_size = 128
        
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        
        pos_init = (.5,.5,.5)
        mean = torch.tensor(pos_init)  # Mean (center) of the Gaussian in 3D
        covariance_matrix = torch.eye(3) * .005
        distribution = torch.distributions.MultivariateNormal(mean, covariance_matrix)
        
        self.init_points = distribution.sample((self.num_gaussians,)).to(device=device)
        
        self.offset_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5
        
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1, full=False):
        scaling = self.get_scaling

        if self.opts.isotropic_scaling:
            scaling = scaling.repeat(1, 3)

        if full:
            return build_symmetric(self.covariance_activation(scaling, scaling_modifier, self._rotation))
        else:
            return self.covariance_activation(scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def forward(self, images):
        """
        Arguments:
            batch: 
                A tuple of (input_view, all_views) where all_views is a list of instances 
                of the class CameraView.

                CameraView has the following properties:
                    - cam_data
                    - scene_rgb
                    - hand_rgb
                    - obj_rgb
                    - scene_mask
                    - hand_mask
                    - hand_mask_with_obj_interaction
                    - obj_mask
                    - scene_depth
                    - hand_depth
                    - obj_depth
                    - scene_normals
                    - hand_normals
                    - obj_normals
                    
                Note: Each attribute supports lazy loading, and as such loads the image into memory when called.
        """
        B, H, W, C = images.shape # [B, V?, H, W, C]
        images = images.permute(0,3,1,2)
        images = images.view(B, C, H, W)
        x = self.unet(images)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.reshape(B, -1, 14).permute(0,2,1)
        x = self.compression_layer(x)
        x = x.permute(0,2,1)
        
        pos = self.init_points+self.offset_act(x[..., 0:3])
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) 
        
        return gaussians
    
        
