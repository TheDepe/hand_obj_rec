import os
import torch
import numpy as np
import cv2 as cv2
import os
import yaml
import trimesh
import torch
import matplotlib.pyplot as plt
import numpy as np
import pymeshlab
import math

import dataclasses

import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


import plotly.graph_objs as go

def list_dirs(root_dir):
    # List only directories in the specified root_dir
    return [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

def apply_mask(img, mask):
    if not torch.is_tensor(img):
        img = torch.tensor(np.array(img))
        
    if mask.shape != img.shape[:2]:
        raise ValueError("Mask size must match the image size")
    return (img * mask.unsqueeze(2)).to(dtype=torch.uint8)

def plot_renderings(CameraView):
    import matplotlib.pyplot as plt
    # List of all the properties to plot
    property_names = [
        "scene_rgb", "hand_rgb", "obj_rgb",
        "scene_mask", "hand_mask", "hand_mask_with_obj_interaction", "obj_mask",
        "scene_depth", "hand_depth", "obj_depth",
        "scene_normals", "hand_normals", "obj_normals"
    ]
    
    # Number of properties
    num_properties = len(property_names)
    
    # Create a grid layout for plotting (assume a 3x4 grid for 12 images)
    cols = 4
    rows = (num_properties // cols) + (num_properties % cols > 0)
    
    # Create a matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    # Loop through each property and plot the corresponding image
    for i, prop_name in enumerate(property_names):
        # Get the image using the property
        img = getattr(CameraView, prop_name)  # Dynamically access the property
        
        # Display the image in the subplot
        axes[i].imshow(img)
        axes[i].set_title(prop_name)
        axes[i].axis('off')  # Hide axis
    
    # Remove empty subplots if the number of properties is not a perfect grid
    for i in range(num_properties, len(axes)):
        axes[i].axis('off')
    
    # Show the plot
    #plt.tight_layout()
    plt.show()
    
    


def plot_3d(pcd, joints=None):
    if not isinstance(pcd, np.ndarray):
        pcd = pcd.detach().cpu().numpy()
    
    if joints is not None and not isinstance(joints, np.ndarray):
        joints = joints.detach().cpu().numpy()

    # Extract coordinates for pcd
    x_pcd = pcd[:, 0]
    y_pcd = pcd[:, 1]
    z_pcd = pcd[:, 2]

    # Initialize traces list with pcd points
    traces = []

    # Add pcd points trace
    traces.append(go.Scatter3d(
        x=x_pcd, y=y_pcd, z=z_pcd,
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.1 if joints is not None else 0.5)
    ))

    if joints is not None:
        # Extract coordinates for joints
        x_joints = joints[:, 0]
        y_joints = joints[:, 1]
        z_joints = joints[:, 2]

        # Add joints points trace
        traces.append(go.Scatter3d(
            x=x_joints, y=y_joints, z=z_joints,
            mode='markers',
            marker=dict(size=5, color='red', opacity=1.0)
        ))

    # Calculate the ranges for each axis
    x_range = max(pcd[:, 0].max(), joints[:, 0].max() if joints is not None else float('-inf')) - \
              min(pcd[:, 0].min(), joints[:, 0].min() if joints is not None else float('inf'))
    y_range = max(pcd[:, 1].max(), joints[:, 1].max() if joints is not None else float('-inf')) - \
              min(pcd[:, 1].min(), joints[:, 1].min() if joints is not None else float('inf'))
    z_range = max(pcd[:, 2].max(), joints[:, 2].max() if joints is not None else float('-inf')) - \
              min(pcd[:, 2].min(), joints[:, 2].min() if joints is not None else float('inf'))

    # Find the maximum range
    max_range = max(x_range, y_range, z_range)

    # Calculate the midpoints of each axis
    x_mid = (pcd[:, 0].max() + pcd[:, 0].min()) / 2
    y_mid = (pcd[:, 1].max() + pcd[:, 1].min()) / 2
    z_mid = (pcd[:, 2].max() + pcd[:, 2].min()) / 2

    # Set the limits for each axis based on the maximum range
    x_lim = [x_mid - max_range / 2, x_mid + max_range / 2]
    y_lim = [y_mid - max_range / 2, y_mid + max_range / 2]
    z_lim = [z_mid - max_range / 2, z_mid + max_range / 2]

    layout = go.Layout(
        title='3D Scatter Plot',
        scene=dict(
            xaxis=dict(range=x_lim, title='X Axis'),
            yaxis=dict(range=y_lim, title='Y Axis'),
            zaxis=dict(range=z_lim, title='Z Axis'),
            aspectmode='cube'  # Ensures the same scaling for all axes
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def generate_random_points_around_means(means, num_points_per_mean, std_dev=0.01):
    """
    Generate random 3D points around given means.
    
    Parameters:
    - means (torch.Tensor): A tensor of shape (n, 3) representing n 3D means.
    - num_points_per_mean (int): Number of random points to generate around each mean.
    - std_dev (float, optional): Standard deviation of the normal distribution. Default is 1.0.
    
    Returns:
    - points (torch.Tensor): A tensor containing the generated points of shape (n * num_points_per_mean, 3).
    """
    all_points = []
    
    for mean in means:
        # Generate random points around the current mean
        points = mean + std_dev * torch.randn(num_points_per_mean, 3)
        all_points.append(points)
    
    # Concatenate all points into a single tensor
    all_points_tensor = torch.cat(all_points, dim=0)
    
    return all_points_tensor

def axis_angle_to_rotation(pose: torch.Tensor, translation: torch.Tensor = None) -> torch.Tensor:
    """
    Convert a batch of pose vectors in axis-angle representation to rotation matrices.
    
    Args:
    - pose (torch.Tensor): A tensor of shape (batch_size, 48) representing the axis-angle rotations for 16 joints.
    
    Returns:
    - torch.Tensor: A tensor of shape (batch_size, 16, 3, 3) representing the rotation matrices for each joint.
    """
    batch_size = pose.shape[0]
    
    num_joints = 16  # since 48 / 3 = 16
    pose_reshaped = pose.view(batch_size, num_joints, 3)  # Reshape to (batch_size, 16, 3)
    if translation == None:
        translation = np.zeros_like(pose_reshaped.detach().cpu().numpy())
        
    elif translation.shape != pose_reshaped.shape:
        # reshape translation
        translation = translation.view(batch_size, num_joints, 3)
        
    # Convert to numpy for compatibility with scipy
    pose_np = pose_reshaped.detach().cpu().numpy()
    
    # Convert each axis-angle representation to a rotation matrix
    r = np.array([R.from_rotvec(pose).as_matrix() for pose in pose_np])  # Reshape to (batch_size * num_joints, 3)
    rotation_matrices_np = r
    rotation_matrices_np = rotation_matrices_np.reshape(batch_size, num_joints, 3, 3)
    
    mono_rotation_matrices = np.tile(np.eye(4), (batch_size, num_joints,1,1))
    # Insert rotation matrices
    mono_rotation_matrices[:, :,:3,:3] = rotation_matrices_np
    # add Translation
    mono_rotation_matrices[:, :,:3,3] = translation
    
    # Convert back to PyTorch tensor
    rotation_matrices = torch.tensor(mono_rotation_matrices, dtype=pose.dtype, device=pose.device)
    
    return rotation_matrices


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def homo(points):
    return torch.nn.functional.pad(points, (0, 1), value=1)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def get_scene_extent(cam_centers):
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    radius = diagonal * 1.1
    return radius

def to_tensor(var, dtype=torch.float32, device=torch.device("cpu")):
    if is_dataclass(var):
        for key, value in var.__dict__.items():
            if value is not None:
                setattr(var, key, to_tensor(value, dtype, device))
    elif is_numpy(var):
        try:
            if len(var.shape) > 0:
                var = attach(torch.tensor(var, dtype=dtype), device)
        except:
            pass
    elif is_list(var):
        try:
            var = attach(torch.tensor(var, dtype=dtype), device)
        except:
            pass
    elif is_tensor(var):
        try:
            var = attach(var, device)
        except:
            pass
    elif is_dict(var):
        try:
            var = dict_to_tensor(var, dtype, device)
        except:
            pass
    return var

def detach(tensor):
    return tensor.detach().cpu()


def attach(tensor, device=torch.device("cpu")):
    return tensor.to(device)

def dict_to_tensor(var, dtype=torch.float32, device=torch.device("cpu")):
    for key, value in var.items():
        if value is not None:
            var[key] = to_tensor(value, dtype, device)
    return var

def is_tensor(var):
    return isinstance(var, torch.Tensor)


def is_numpy(var):
    return isinstance(var, np.ndarray)


def is_list(var):
    return isinstance(var, list)


def is_dict(var):
    return isinstance(var, dict)


def is_dataclass(var):
    return dataclasses.is_dataclass(var)


def is_string(var):
    return isinstance(var, str)


def is_dict(var):
    return isinstance(var, dict)


def get_opengl_camera_attributes(
    K, extrins, width, height, zfar=100.0, znear=0.01, resize_factor=1.0
):
    K[..., :2, :] = K[..., :2, :] * resize_factor
    width = int(width * resize_factor + 0.5)
    height = int(height * resize_factor + 0.5)
    fovx = focal2fov(K[0, 0], width)
    fovy = focal2fov(K[1, 1], height)
    extrins = np.concatenate([extrins, np.array([[0, 0, 0, 1]])], axis=0)
    world_view_transform = np.transpose(extrins)
    projection_matrix = np.transpose(
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
    )
    full_proj_transform = np.matmul(world_view_transform, projection_matrix)
    camera_center = np.linalg.inv(world_view_transform)[3, :3]

    out = {
        "width": width,
        "height": height,
        "fovx": fovx,
        "fovy": fovy,
        "K": K,
        "extr": extrins,
        "world_view_transform": world_view_transform,
        "projection_matrix": projection_matrix,
        "full_proj_transform": full_proj_transform,
        "camera_center": camera_center,
    }
    return out

    
def rotation_matrix_to_mano_transforms(rot_mats: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """
    Arguments: 
        rot_mats: tensor of BxJx3x3 
        joints: tensor of BxJx3 (mano4DGS.joints)
        parents: tensor (mano4DGS.mano_data['kentree_table'][0])
    """
    
    joints = joints.unsqueeze(-1) # BxNx3x1 (extra 1 dimension for fun)
    rel_joints = joints.clone()
    #parents = self.mano_data['kintree_table'][0] # parents   # CAN BE ACCESSES VIA SELF
    rel_joints[:,1:] -= joints[:, parents[1:]]
    
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)


    transform_chain = [transforms_mat[:,0]]
    for i in range(1, parents.shape[0]):
        # subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:,i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3,0,0,0,0,0,0,0])

    return rel_transforms
    

def dump_points(points, path, colors=None):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    pc = trimesh.PointCloud(points)
    if colors is not None:
        pc.visual.vertex_colors = colors[..., :4]
    pc.export(path)

def visualize_skin_weights(skin_weights):
    if isinstance(skin_weights, torch.Tensor):
        skin_weights = skin_weights.numpy()

    cmap = plt.get_cmap("tab20c")
    num_bones = skin_weights.shape[-1]
    color_values = np.linspace(0, 1, num_bones)
    color_map = cmap(color_values)
    indices = np.argmax(skin_weights, axis=-1)
    colors = color_map[indices]
    return colors

# Not used 
def extract_filename_without_extension(filepath):
    # Get the base name of the file (e.g., "fruitbowl.avi")
    base_name = os.path.basename(filepath)
    # Split the base name into name and extension (e.g., "fruitbowl" and ".avi")
    name, _ = os.path.splitext(base_name)
    return name

# Not used
def get_unique_folder_name(base_path):
    if not os.path.exists(base_path):
        return base_path
    
    counter = 1
    new_path = f"{base_path}_{counter}"
    while os.path.exists(new_path) and os.listdir(new_path):
        counter += 1
        new_path = f"{base_path}_{counter}"
    
    return new_path

# Not used
def split_frame(videopath, fps=30, verbose=False):
    if verbose:
        print(f'processing {videopath} .....')
    # filepath = os.path.dirname(videopath)
    # videoname = os.path.basename(videopath)[:-4]
    if not os.path.exists(videopath):
        raise FileNotFoundError('{} not exists!'.format(videopath))

    vc = cv2.VideoCapture(videopath)

    path = videopath.rsplit('.', 1)[0]+"_" + str(fps) +"fps_images"
    
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        print(f'Images alread exist at {path}.')
        return path
    
    #path = get_unique_folder_name(path)
    if not os.path.exists(path):   
        os.mkdir(path)
        
    original_fps = vc.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps)

    Frame = 0
    saved_frame = 0
    while True:
        rval, frame = vc.read()
        if not rval:
            break

        if Frame % frame_interval == 0:
            if 'MOV' not in videopath:
                cv2.imencode('.jpg', frame)[1].tofile(f'{path}/{str(saved_frame).zfill(6)}.jpg')
            else:
                cv2.imencode('.jpg', frame[::-1, ::-1])[1].tofile(f'{path}/{str(saved_frame).zfill(6)}.jpg')
            saved_frame += 1
        if verbose:
            print(f'Processing Frame {Frame}...')
        Frame += 1

    vc.release()
    if verbose:
        print('Saved to', path)
    return path



def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)



def build_symmetric(L):
    cov = torch.zeros((L.shape[0], 3, 3), dtype=L.dtype, device=L.device)
    cov[:, 0, 0] = L[:, 0]
    cov[:, 0, 1] = L[:, 1]
    cov[:, 0, 2] = L[:, 2]
    cov[:, 1, 1] = L[:, 3]
    cov[:, 1, 2] = L[:, 4]
    cov[:, 2, 2] = L[:, 5]

    cov[:, 1, 0] = L[:, 1]
    cov[:, 2, 0] = L[:, 2]
    cov[:, 2, 1] = L[:, 4]
    return cov


def build_rotation(r, device="cuda"):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r, device="cuda"):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=device)
    R = build_rotation(r, device=device)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

def update_mask_based_on_outliers(xyz, prob=0.5, neighbors=128):
    pts_path = "./noisy.ply"
    dump_points(xyz, pts_path)
    mask = remove_outliers(pts_path, prob, neighbors)
    return mask

def remove_outliers(pts_path, prob, neighbors):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(pts_path)
    ms.compute_selection_point_cloud_outliers(propthreshold=prob, knearest=neighbors)
    return ms.current_mesh().vertex_selection_array()

