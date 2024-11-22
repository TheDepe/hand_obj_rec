import os
import cv2
import bpy
import joblib
import mathutils
import numpy as np

from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Cameras:
    cam_name: np.ndarray
    K: np.ndarray
    distK: np.ndarray
    dist: np.ndarray
    extr: np.ndarray
    
    def __getitem__(self, key):
        if isinstance(key, str):
            idx = np.where(self.cam_name == key)[0][0]
        elif isinstance(key, int):
            idx = key
        else:
            raise TypeError("Key must be either an integer or a string representing the camera name.")

        new_dict = {}
        for key, value in self.__dict__.items():
            new_dict[key] = value[idx]
        return Cameras(**new_dict)

def setup_scene():
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    collection = bpy.data.collections.get('Collection')
    bpy.data.collections.remove(collection)
    bpy.context.view_layer.update()  # Force scene update after clearing

    # Add cameras to the scene
    cameras = get_cam_data("/home/perrettde/Documents/thesis/DATA/Manus_Data/optim_params.txt")
    create_cameras(cameras)

    
    # Optional: Adjust render settings if necessary
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720

    bpy.context.view_layer.update()
def create_cameras(cameras):
    # Extract camera extrinsics (rotation and location)
    
    cam_collection_name = "CameraCollection"
    cam_collection = bpy.data.collections.new(cam_collection_name)
    bpy.context.scene.collection.children.link(cam_collection)
    
    cam_names = cameras.cam_name
    SCALE = 1
    cam_centers = []
    
    rotation1 = mathutils.Matrix.Rotation(np.radians(0), 4, 'X')
    rotation2 = mathutils.Matrix.Rotation(np.radians(0), 4, 'Y')
    grid_rotation = rotation2 @ rotation1

    for cam_name in cam_names:
        camera = cameras[cam_name]
        # Create the camera object
        bpy.ops.object.camera_add()
        ob = bpy.context.object
        ob.name = cam_name
        cam = ob.data
        cam.name = "_" + cam_name
        cam.type = 'PERSP'
        K = camera.K

        RT = np.vstack([camera.extr, np.array([0, 0, 0, 1])])

        R_world2cv = RT[:3, :3]
        T_world2cv = RT[:3, 3]

        scene = bpy.context.scene
        sensor_width_in_mm = K[1, 1] * K[0, 2] / (K[0, 0] * K[1, 2])
        sensor_height_in_mm = 1  # Doesn't matter apparently
        resolution_x_in_px = 1280
        resolution_y_in_px = 720
        pixel_aspect_ratio = resolution_x_in_px / resolution_y_in_px

        s_u = resolution_x_in_px / sensor_width_in_mm
        s_v = resolution_y_in_px / sensor_height_in_mm

        f_in_mm = K[0, 0] / s_u

        scene.render.resolution_x = int(resolution_x_in_px / SCALE)
        scene.render.resolution_y = int(resolution_y_in_px / SCALE)
        scene.render.resolution_percentage = SCALE * 100

        R_bcam2cv = mathutils.Matrix(
            ((1, 0, 0),
             (0, -1, 0), #-1
             (0, 0, -1)) #-1
        )
        R_cv2world = R_world2cv.T

        rotation = mathutils.Matrix(R_cv2world @ R_bcam2cv)
        location = mathutils.Vector(-R_cv2world @ T_world2cv)
        sensor_fit = get_sensor_fit(cam.sensor_fit,
                                    scene.render.pixel_aspect_x * resolution_x_in_px,
                                    scene.render.pixel_aspect_y * resolution_y_in_px
                                    )
        if sensor_fit == 'HORIZONTAL':
            view_fac_in_px = resolution_x_in_px
        else:
            view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px

        cam.lens = f_in_mm
        cam.lens_unit = 'MILLIMETERS'
        cam.sensor_width = sensor_width_in_mm

        # Apply the 90-degree rotation to the whole camera grid
        # Using matrix multiplication: rotated_matrix = global_rotation_matrix @ local_matrix
        ob.matrix_world = grid_rotation @ (mathutils.Matrix.Translation(location) @ rotation.to_4x4())
        cam_centers.append(ob.location)
        cam.shift_x = (-(K[0, 2] - (resolution_x_in_px / 2)) / view_fac_in_px)
        cam.shift_y = ((K[1, 2] - (resolution_y_in_px / 2)) * pixel_aspect_ratio / view_fac_in_px)

        ob.scale = (0.05, 0.05, 0.05)
        scene.camera = ob
        bpy.context.view_layer.update()
        # Add the camera to the new camera collection
        bpy.context.scene.collection.objects.unlink(ob)  # Unlink it from the scene collection
        cam_collection.objects.link(ob)  # Link it to the camera collection
        
    #grid_center = np.mean(cam_centers, axis=0)
    #for camera in [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']:
    #    camera.location -= mathutils.Vector(grid_center)
        
        
def read_params(params_path, legacy_calib=False):
    if legacy_calib:
        params = joblib.load(params_path)["0"]
    else:
        params = np.loadtxt(
            params_path,
            dtype=[
                ("cam_id", int),
                ("width", int),
                ("height", int),
                ("fx", float),
                ("fy", float),
                ("cx", float),
                ("cy", float),
                ("k1", float),
                ("k2", float),
                ("p1", float),
                ("p2", float),
                ("cam_name", "<U22"),
                ("qvecw", float),
                ("qvecx", float),
                ("qvecy", float),
                ("qvecz", float),
                ("tvecx", float),
                ("tvecy", float),
                ("tvecz", float),
            ],
        )
        params = np.sort(params, order="cam_name")

    return params

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def get_extr(param, legacy_calib=False):
    if legacy_calib:
        extr = param["extrinsics_opencv"]
    else:
        qvec = [param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]]
        tvec = np.asarray([param["tvecx"], param["tvecy"], param["tvecz"]])
        r = qvec2rotmat(qvec)
        extr = np.vstack([np.hstack([r, tvec[:, None]]), np.zeros((1, 4))])
        extr[3, 3] = 1
        extr = extr[:3]

    return extr

def get_intr(param, undistort=False, legacy_calib=False):
    intr = np.eye(3)
    if legacy_calib:
        intr[0, 0] = param["K"][0]
        intr[1, 1] = param["K"][1]
        intr[0, 2] = param["K"][2]
        intr[1, 2] = param["K"][3]
        dist = np.asarray([0, 0, 0, 0])
    else:
        intr[0, 0] = param["fx_undist" if undistort else "fx"]
        intr[1, 1] = param["fy_undist" if undistort else "fy"]
        intr[0, 2] = param["cx_undist" if undistort else "cx"]
        intr[1, 2] = param["cy_undist" if undistort else "cy"]

        # TODO: Make work for arbitrary dist params in opencv
        dist = np.asarray([param["k1"], param["k2"], param["p1"], param["p2"]])

    return intr, dist
def get_undistort_params(intr, dist, img_size):
    new_intr = cv2.getOptimalNewCameraMatrix(
        intr, dist, img_size, alpha=0, centerPrincipalPoint=True
    )
    return new_intr

def get_cam_data(cam_path, width = 1280, height = 720): 
    cameras = read_params(cam_path)
    d_dict = defaultdict(list)
    for idx, cam in enumerate(cameras):
        extr = get_extr(cam)
        K, dist = get_intr(cam)
        cam_name = cam["cam_name"]
        new_K, roi = get_undistort_params(K, dist, (width, height))
        new_K = new_K.astype(np.float32)
        extr = extr.astype(np.float32)
        d_dict['cam_name'].append(cam_name)
        d_dict['dist'].append(dist)
        d_dict['distK'].append(K)
        d_dict['extr'].append(extr)
        d_dict['K'].append(new_K)

    for k, v in d_dict.items():
        d_dict[k] = np.stack(v, axis=0)

    return Cameras(**d_dict)


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


if __name__ == '__main__':
    setup_scene()
    bpy.ops.wm.save_mainfile(filepath=os.path.join('outputs',"scene.blend"))