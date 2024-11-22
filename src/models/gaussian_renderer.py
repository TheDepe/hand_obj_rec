import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

import kiui


class GaussianRenderer:
    def __init__(self, opt):
        
        self.opt = opt
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        
        # intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1
        
    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, sub_pixel_offset=None, scale_modifier=1.0):

        device = gaussians.device
        B, V = cam_view.shape[:2]

        # loop of loop...
        images = []
        normals = []
        depths = []
        masks = []
        distortions = []
      
        for b in range(B):

            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float()

            for v in range(V):
                
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=720,
                    image_width=1280,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    #kernel_size=0.0,
                    #subpixel_offset= torch.Tensor([]) if sub_pixel_offset is None else None,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                rendered_image, radii = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_rgb = rendered_image[:3, :, :].clamp(0, 1)
                #rendered_normal = rendered_image[3:6, :, :]
                #rendered_depth = rendered_image[6:7, :, :]
                #rendered_mask = rendered_image[7:8, :, :]
                #rendered_distortion = rendered_image[8:9, :, :]

                rendered_rgb = rendered_rgb.clamp(0, 1)

                images.append(rendered_rgb)
                #normals.append(rendered_normal)
                #depths.append(rendered_depth)
                #masks.append(rendered_mask)
                #distortions.append(rendered_distortion)

        images = torch.stack(images, dim=0).view(B, V, 3, self.opt.output_size_y, self.opt.output_size_x)
        #normals = torch.stack(normals, dim=0).view(B, V, 3, self.opt.output_size_x, self.opt.output_size_y)
        #depths = torch.stack(depths, dim=0).view(B, V, 1, self.opt.output_size_x, self.opt.output_size_y)
        #masks = torch.stack(masks, dim=0).view(B, V, 1, self.opt.output_size_x, self.opt.output_size_y)
        #distortions = torch.stack(distortions, dim=0).view(B, V, 1, self.opt.output_size_x, self.opt.output_size_y)

        return {
            "image": images,
            #"normal": normals,
            #"depth": depths,
            #"mask": masks,
            #"distortion": distortions 
        }

    def integrate(self, points3D, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, sub_pixel_offset=None, scale_modifier=1.0):

        device = gaussians.device
        B, V = cam_view.shape[:2]

        images = []
        normals = []
        depths = []
        masks = []
        distortions = []

        alpha_integrated_all = []
        color_integrated_all = []

        for b in range(B):
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() 

            for v in range(V):
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.opt.output_size_y,
                    image_width=self.opt.output_size_x,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    kernel_size=0.0,
                    subpixel_offset= torch.Tensor([]) if sub_pixel_offset is None else None,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                rendered_image, alpha_integrated, color_integrated, radii = rasterizer.integrate(
                    points3D = points3D,
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_rgb = rendered_image[:3, :, :].clamp(0, 1)
                #rendered_normal = rendered_image[3:6, :, :]
                #rendered_depth = rendered_image[6:7, :, :]
                #rendered_mask = rendered_image[7:8, :, :]
                #rendered_distortion = rendered_image[8:9, :, :]

                rendered_rgb = rendered_rgb.clamp(0, 1)

                images.append(rendered_rgb)
                #normals.append(rendered_normal)
                #depths.append(rendered_depth)
                #masks.append(rendered_mask)
                #distortions.append(rendered_distortion)

                alpha_integrated_all.append(alpha_integrated)
                color_integrated_all.append(color_integrated)

        images = torch.stack(images, dim=0).view(B, V, 3, self.opt.output_size_y, self.opt.output_size_x)
        #normals = torch.stack(normals, dim=0).view(B, V, 3, self.opt.output_size_x, self.opt.output_size_y)
        #depths = torch.stack(depths, dim=0).view(B, V, 1, self.opt.output_size_x, self.opt.output_size_y)
        #masks = torch.stack(masks, dim=0).view(B, V, 1, self.opt.output_size_x, self.opt.output_size_y)
        #distortions = torch.stack(distortions, dim=0).view(B, V, 1, self.opt.output_size_x, self.opt.output_size_y)

        alpha_integrated_all = torch.stack(alpha_integrated_all, dim=0)
        color_integrated_all = torch.stack(color_integrated_all, dim=0)

        return {
            "image": images, 
            #"normal": normals, 
            #"depth": depths, 
            #"mask": masks, 
            #"distortion": distortions, 
            'alpha_integrated': alpha_integrated_all, 
            'color_integrated': color_integrated_all, 
        }
    

    @torch.no_grad()
    def evaluate_alpha(self, points, gaussians, cam_view, cam_view_proj, cam_pos):
        final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) 

        with torch.no_grad():
            ret = self.integrate(points, gaussians, cam_view, cam_view_proj, cam_pos, bg_color)
            alpha_integrated = ret["alpha_integrated"] 
            alpha_integrated = alpha_integrated.squeeze(0) 
            for viewidx in range(alpha_integrated.shape[0]):
                alpha_integrated_view = alpha_integrated[viewidx]
                final_alpha = torch.min(final_alpha, alpha_integrated_view) 
            alpha = 1 - final_alpha
        return alpha

    @staticmethod
    def alpha_to_sdf(alpha):    
        sdf = alpha - 0.5
        sdf = sdf[None]
        return sdf
    

    def save_ply(self, gaussians, path, compatible=True):
        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() 

        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() 

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians