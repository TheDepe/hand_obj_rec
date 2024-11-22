import torch
import math
import torch.nn
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

import cv2
from utils.utils import *
from utils.sh_utils import eval_sh

def render_gaussians(
    posed_means,
    posed_cov,
    cano_means,
    cano_features,
    cano_opacity,
    camera,
    bg_color,
    colors_precomp=None,
    sh_degree=3,
    tf=None,
    device=torch.device("cuda"),
):
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            posed_means, dtype=posed_means.dtype, requires_grad=True, device=device
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    tanfovx = math.tan(camera.fovx * 0.5)
    tanfovy = math.tan(camera.fovy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1,
        viewmatrix=camera.world_view_transform.to(device),
        projmatrix=camera.full_proj_transform.to(device),
        sh_degree=sh_degree,
        campos=camera.camera_center.to(device),
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = cano_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.

    if colors_precomp is None:
        colors_precomp = calculate_colors_from_sh(
            posed_means, cano_features, cano_means, camera, sh_degree, tf
        )
 
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=posed_means,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=posed_cov,
    )

    rendered_image = torch.permute(rendered_image, (1, 2, 0))


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def project_points(points, K, extrin):
    BS = points.shape[0]
    points = homo(points)
    P = torch.matmul(K, extrin)
    P = torch.tile(P.unsqueeze(0), (BS, points.shape[1], 1, 1))
    points = torch.einsum("BNij,BNj->BNi", P, points)
    points = points / points[..., 2:]
    return points[..., :2]



def calculate_colors_from_sh(
    posed_means, cano_features, cano_means, camera, sh_degree, tf
):
    shs_view = cano_features.transpose(1, 2).view(-1, 3, (sh_degree + 1) ** 2)
    camera_center = camera.camera_center.repeat(cano_features.shape[0], 1)

    if tf is not None:
        cam_inv = torch.einsum(
            "nij, nj->ni", torch.linalg.inv(tf), homo(camera_center)
        )[..., :3]
        dir_pp_inv = cano_means - cam_inv
        dir_pp_normalized = dir_pp_inv / dir_pp_inv.norm(dim=1, keepdim=True)
    else:
        dir_pp = posed_means - camera_center
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)

    sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    return colors_precomp


def update_learning_rate(optimizer, model, global_step):
    """Learning rate scheduling per step"""
    for param_group in optimizer.param_groups:
        if param_group["name"] == "xyz":
            lr = model.xyz_scheduler_args(global_step)
            param_group["lr"] = lr
            return optimizer
def get_points_outside_mask(camera, points, mask, keypoints=None, dilate=False):
    K = camera.K
    extr = camera.extr
    if len(K.shape) == 3:
        K = K[0]
    if len(extr.shape) == 3:
        extr = extr[0]

    if len(mask.shape) == 4:
        mask = mask[0]

    # Dilate Mask
    if dilate:
        mask = dilate_mask(mask[..., 0]).unsqueeze(-1).int()

    p2d = project_points(points[None], K, extr[:3, :4])[0]

    pts_x = torch.clamp(p2d[..., 0], 0, mask.shape[1] - 1).int()
    pts_y = torch.clamp(p2d[..., 1], 0, mask.shape[0] - 1).int()
    mask = ~mask.bool()
    mask_value = mask[pts_y, pts_x]

    # If any of the keypoint is outside the mask, ignore mask_value
    if keypoints is not None:
        k2d = project_points(keypoints[None], K, extr[:3, :4])[0]
        pts_x = torch.clamp(k2d[..., 0], 0, mask.shape[1] - 1).int()
        pts_y = torch.clamp(k2d[..., 1], 0, mask.shape[0] - 1).int()
        keypt_mask_value = mask[pts_y, pts_x]
        if torch.any(keypt_mask_value):
            mask_value = torch.zeros_like(mask_value)

    """ Visualization debug code
    mask_value = torch.zeros_like(mask_value)
    p2d = p2d[mask_value[..., 0].bool()]
    mask = mask.repeat(1, 1, 3)
    img = to_numpy(mask) * 255
    img = np.ascontiguousarray(img).astype(np.uint8)
    
    ## visualize these points using OpenCV
    for idx in range(p2d.shape[0]):
        x = int(p2d[idx][0])
        y = int(p2d[idx][1])
        img = cv2.circle(img, (x, y), 1, (0, 255, 0), 1)
    dump_image(img, './proj_points.png')
    """

    return mask_value

def dilate_mask(mask, kernel_size=11):
    kernel = torch.ones(
        (kernel_size, kernel_size), dtype=torch.float32, device=mask.device
    )
    mask = mask.float()
    mask = torch.nn.functional.conv2d(
        mask.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=kernel_size // 2,
    )
    mask = mask.squeeze(0).squeeze(0)
    mask = mask > 0
    return mask


def density_update(model, pred, extent, global_step, bg_color, mask_to_prune=None):
    update_optimizer = False
    with torch.no_grad():
        if mask_to_prune is not None:
            model.prune_points(mask_to_prune)
            torch.cuda.empty_cache()
            update_optimizer = True

            #print(f"Removing pts : {mask_to_prune.sum()}")
        else:
            viewspace_point_tensor = pred["viewspace_points"]
            visibility_filter = pred["visibility_filter"]
            radii = pred["radii"]
            cameras_extent = extent

            # Densification
            if global_step < model.densify_until_step:
                # Keep track of max radii in image-space for pruning
                model.max_radii2D[visibility_filter] = torch.max(
                    model.max_radii2D[visibility_filter], radii[visibility_filter]
                )

                model.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if (
                    global_step > model.densify_from_step
                    and global_step % model.densification_interval == 0
                ):
                    size_threshold = (
                        model.size_threshold
                        if global_step > model.opacity_reset_interval
                        else None
                    )

                    clean_outliers = global_step == model.remove_outliers_step
                    model.densify_and_prune(
                        model.densify_grad_threshold,
                        model.min_opacity_threshold,
                        cameras_extent,
                        size_threshold,
                        clean_outliers,
                    )
                    #print("gaussians changed to ", model.get_xyz.shape[0])
                    update_optimizer = True

                if global_step % model.opacity_reset_interval == 0 or (
                    (bg_color == "white")
                    and global_step == model.densify_from_step
                ):
                    if global_step != 0:
                        model.reset_opacity()
                        update_optimizer = True
    return update_optimizer