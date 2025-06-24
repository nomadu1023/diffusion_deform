from nets.module import GaussianRenderer
from utils.general_utils import get_expon_lr_func, build_rotation, quat_mult, point_cloud_to_image

import os.path as osp
from utils.general_utils import get_expon_lr_func
import torch.nn as nn
from scipy.spatial.transform import Rotation 

# Copyright (c) 2024 DreamScene4D and affiliated authors.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import glob
import tqdm
import json
import scipy
import pickle
import numpy as np

import argparse
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from torchmetrics import PearsonCorrCoef

from cameras import orbit_camera, OrbitCamera, MiniCam
from utils.general_utils import safe_normalize
from gs_renderer_4d import Renderer
from utils.flow_utils import run_flow_on_images
from gmflow.gmflow.gmflow import GMFlow

from PIL import Image
from torchvision.transforms.functional import center_crop
from scipy.spatial.transform import Rotation as R



def world_initialize(self):
 

    # Initialize translation and scale parameters for all frames
    self.translations = nn.Parameter(
        torch.zeros((self.vid_length, 3), dtype=torch.float32, device=self.device),
        requires_grad=False
    )
    
    self.scales = nn.Parameter(
        torch.zeros((self.vid_length,), dtype=torch.float32, device=self.device),
        requires_grad=False
    )

    self.rotations = nn.Parameter(
        torch.zeros((self.vid_length, 4), dtype=torch.float32, device=self.device),
        requires_grad=False
    )

    
    # Load camera intrinsics from the calibration file (cameras.txt)
    intrinsics = {}  # Dictionary to map camera ID to intrinsic matrix
    intrinsic_path =  self.opt.intrinsics_file
    if intrinsic_path is None:
        # If not explicitly provided, assume cameras.txt is alongside cam_params.pkl
        intrinsic_path = osp.join(osp.dirname(self.args.cam_param_path), "cameras.txt")
    if os.path.exists(intrinsic_path):
        with open(intrinsic_path, 'r') as f:
            for line in f:
                if line.strip().startswith('#') or line.strip() == '':
                    continue  # skip comments/empty lines
                parts = line.strip().split()
                if len(parts) >= 8:
                    cam_id = int(parts[0])
                    model = parts[1]
                    width = float(parts[2]); height = float(parts[3])
                    fx = float(parts[4]); fy = float(parts[5])
                    cx = float(parts[6]); cy = float(parts[7])
                    # Construct intrinsic matrix K for this camera
                    K = np.array([[fx, 0.0, cx],
                                [0.0, fy, cy],
                                [0.0, 0.0, 1.0]], dtype=np.float32)
                    intrinsics[cam_id] = K
    else: 
        print(f"Warning: Intrinsic file not found at {intrinsic_path}.")
    
    # If depth files are available, prepare a list of depth file paths
    depth_dir = self.opt.depth_dir
    depth_files = []
    if depth_dir is not None and os.path.isdir(depth_dir):
        depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.npy')))
        if len(depth_files) != self.vid_length:
            print("Warning: Number of depth files does not match number of frames.")
    # If depth_dir is not provided, we will derive depth path per frame from mask filename
    
    for img, mask in zip(self.input_img_list, self.input_mask_list):
        # self.input_img_torch_list_2.append(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device))
        self.input_mask_torch_list.append(torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0))
    
    # Iterate over each frame to compute translation and set scale
    for idx in range(self.vid_length):
        
        # Check if the object is present in this frame (mask has any foreground)
        if torch.sum(self.input_mask_torch_list[idx]) == 0:
            # No object in frame: keep default translation [0,0,0] and scale 1.0
            self.translations.data[idx] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            self.scales.data[idx] = 0.0 # 1.0
            continue
        
        # **1. Get the mask and compute object's centroid in pixel coordinates (obj_cx, obj_cy)**
        mask_np = self.input_mask_list[idx]  # mask as a NumPy array (H x W x 1)
        mask_np = mask_np[:, :, 0]  # reshape to 2D (H x W)
        # Compute centroid: mean of the coordinates of all mask pixels
        ys, xs = np.nonzero(mask_np)  # indices of foreground (object) pixels
        obj_cy = np.mean(ys)  # centroid y-coordinate (row index)
        obj_cx = np.mean(xs)  # centroid x-coordinate (column index)
        # (Alternatively, one could use bounding box center. For example:
        #  min_y, max_y = ys.min(), ys.max()
        #  min_x, max_x = xs.min(), xs.max()
        #  obj_cy = 0.5 * (min_y + max_y)
        #  obj_cx = 0.5 * (min_x + max_x))
        
        # **2. Load the corresponding depth map and get depth value at the centroid**
        depth_value = None
        if depth_files:
            # If depth files list is prepared, use it directly
            depth_path = depth_files[idx]
        else:
            # Derive depth file path from mask/image filename (assuming same base name)
            # Example: if mask filename is "frame_00190_mask.png", depth file is "frame_00190.npy"
            img_name = os.path.basename(self.input_img_list[idx]) if isinstance(self.input_img_list[idx], str) \
                    else None
            mask_name = os.path.basename(self.input_mask_list[idx]) if isinstance(self.input_mask_list[idx], str) \
                        else None
            # Try to use mask filename if available (the mask file path might not be stored as string here)
            base_name = ""
            if mask_name is not None:
                base_name = mask_name.replace('_mask', '').split('.')[0]
            elif img_name is not None:
                base_name = img_name.replace('_rgb', '').split('.')[0]
            depth_path = os.path.join(depth_dir if depth_dir else "", base_name + ".npy")
        try:
            depth_map = np.load(depth_path)
            H, W = depth_map.shape[:2]
            # Round centroid to nearest pixel indices
            px = int(round(obj_cx))
            py = int(round(obj_cy))
            if 0 <= py < H and 0 <= px < W:
                depth_value = float(depth_map[py, px])
            else:
                depth_value = None
        except Exception as e:
            depth_value = None
            print(f"Warning: Could not load depth for frame {idx} at {depth_path}: {e}")
        
        if depth_value is None or depth_value <= 0 or np.isnan(depth_value):
            # If depth is missing or invalid, skip this frame (leave default translation)
            self.translations.data[idx] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            # Use mask-based scale as fallback
            self.scales.data[idx] = self.obj_scale_list[idx] if idx < len(self.obj_scale_list) else 1.0
            continue
        
        
        X_cam = (obj_cx - cx) * depth_value / fx
        Y_cam = (obj_cy - cy) * depth_value / fy
        Z_cam = depth_value
        # print(depth_value)
        point_cam = np.array([X_cam, Y_cam, Z_cam], dtype=np.float32)
        
        # **4. Convert the 3D camera-coordinate point to world coordinates using extrinsics**
        # We need the camera extrinsic (orientation and position). Assume cam_params contains this.
        # Determine orientation (rotation) and position for this frame’s camera.
        R_c2w = None
        t_c2w = None
        
        cam_extrinsic = None
        
        
        if cam_extrinsic is None:
            # If not already retrieved above, get it now (for dict with numeric keys or list).
            if isinstance(self.cam_params, dict) and idx in self.cam_params:
                cam_extrinsic = self.cam_params[idx]
                cam_extrinsic = to_cpu(cam_extrinsic)
                
            elif isinstance(self.cam_params, (list, tuple)) and idx < len(self.cam_params):
                cam_extrinsic = self.cam_params[idx] 
                
                cam_extrinsic = to_cpu(cam_extrinsic)
                
        
        if isinstance(cam_extrinsic, dict):
            # If extrinsic is stored as orientation matrix and position vector
            if 'orientation' in cam_extrinsic and 'pos' in cam_extrinsic:
                R_c2w = np.array(cam_extrinsic['orientation'], dtype=np.float32)
                t_c2w = np.array(cam_extrinsic['pos'], dtype=np.float32)
            if 'R' in cam_extrinsic and 't' in cam_extrinsic:
                # 'R' could be world->cam or cam->world; assume T is camera position if R is cam->world
                R_val = np.array(cam_extrinsic['R'], dtype=np.float32)
                T_val = np.array(cam_extrinsic['t'], dtype=np.float32)
                if R_val.shape == (3, 3):
                    # If R is world->cam, convert to cam->world by transpose
                    R_c2w = R_val.T
                    t_c2w = -R_val.T.dot(T_val.reshape(3))
                else:
                    # If R is already cam->world (orientation), use it directly
                    R_c2w = R_val[:3, :3] if R_val.shape == (4, 4) else R_val
                    t_c2w = T_val.reshape(3)

                initial_R = False
                if initial_R:
                    quat = Rotation.from_matrix(R_val).as_quat()  # (x, y, z, w) format
                    self.rotations.data[idx] = torch.tensor(quat, device=self.device)
                else:
                    self.rotations.data[idx] = torch.tensor([0, 0, 0, 1], device=self.device)


            elif isinstance(cam_extrinsic.get(0), np.ndarray):
                # If stored as an array under numeric keys (less likely), handle similarly
                R_c2w = np.array(cam_extrinsic[0], dtype=np.float32)
                t_c2w = np.array(cam_extrinsic[1], dtype=np.float32) if 1 in cam_extrinsic else None
        elif isinstance(cam_extrinsic, np.ndarray):
            # If extrinsic is a numpy array (e.g., 4x4 or 3x4 matrix)
            if cam_extrinsic.shape == (4, 4):
                R_c2w = cam_extrinsic[:3, :3].astype(np.float32)
                t_c2w = cam_extrinsic[:3, 3].astype(np.float32)
            elif cam_extrinsic.shape == (3, 4):
                R_c2w = cam_extrinsic[:3, :3].astype(np.float32)
                # cam_extrinsic[:3, 3] could be translation in world->cam; assume we stored cam->world
                t_c2w = cam_extrinsic[:3, 3].astype(np.float32)
        
        if R_c2w is None or t_c2w is None:
            print(f"Warning: Incomplete extrinsic for frame {idx}, setting translation to origin.")
            world_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            # Compute world coordinates of the object centroid
            # If R_c2w is rotation from camera to world and t_c2w is camera position in world:
            world_point = R_c2w.dot(point_cam) + t_c2w
            # (If R_c2w, t_c2w are actually world->cam, one would invert, but we assume cam->world here.)
        
        # **5. Set this world coordinate as the translation initialization for the frame**
        self.translations.data[idx] = torch.tensor(world_point, dtype=torch.float32, device=self.device)
        
        
        # **6. Initialize scale for this frame.** 
        # Use the existing mask bounding-box method (obj_scale_list) unless depth suggests otherwise.
        if idx < len(self.obj_scale_list):
            self.scales.data[idx] = self.obj_scale_list[idx]
        else:
            self.scales.data[idx] = 1.0

def to_cpu(obj):
    
    if isinstance(obj, torch.Tensor):
        return obj.cpu()

    # 파이썬 내장 컨테이너 처리
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    if isinstance(obj, set):
        return {to_cpu(v) for v in obj}

    # 기타 사용자 정의 클래스(예: dataclass)나 네임드튜플 등 처리
    if hasattr(obj, "__dict__"):
        # __dict__를 가진 객체면, 얕은 복사 후 속성별로 재귀 적용
        new_obj = obj.__class__.__new__(obj.__class__)
        for k, v in obj.__dict__.items():
            setattr(new_obj, k, to_cpu(v))
        return new_obj

    # 위 조건에 걸리지 않으면 원본 그대로
    return obj



def gaussian_render_flow(
    self,
    viewpoint_cameras,
    cam_param = None,
    render_resolution = None,
    time = None
):
    assert len(viewpoint_cameras) == 2
    curr, prev = viewpoint_cameras
    
    curr_time, prev_time = time
    
    # # Extract 2D and 3D means for current and previous views
    gaussian_2d_pos_curr, gaussian_3d_pos_curr = curr[0]['mean_2d'], curr[1]
    gaussian_2d_pos_prev, gaussian_3d_pos_prev = prev[0]['mean_2d'], prev[1]

    # # Compute 2D flow between views and pad to (dx, dy, 0)
    flow_2d = gaussian_2d_pos_curr - gaussian_2d_pos_prev
    flow_padded = torch.cat([flow_2d, torch.zeros_like(flow_2d[:, 1:])], dim=1)



    # N = gaussian_3d_pos_curr.shape[0]
    # zeros = torch.zeros((N, 4), dtype=flow_2d.dtype, device=flow_2d.device)  # [N, 4]
    # flow_padded = torch.cat([flow_2d, zeros], dim=1)
    
    
    # flow_padded = torch.cat([flow_2d, torch.zeros_like(flow_2d[:, :3])], dim=1)  # [N, 5]
    # # or, more generally:
    # zeros = torch.zeros((gaussian_3d_pos_curr.shape[0], 3), dtype=flow_2d.dtype, device=flow_2d.device)
    # flow_padded = torch.cat([flow_2d, zeros], dim=1)  # [N, 5]
    
    
    # # Build GaussianRenderer assets (mean_3d is in world coordinates)
    # N = gaussian_3d_pos_curr.shape[0]
    # device = gaussian_3d_pos_curr.device; dtype = gaussian_3d_pos_curr.dtype
    # gaussian_assets = {
    #     'mean_3d': gaussian_3d_pos_curr,
    #     'opacity': torch.ones((N, 1), dtype=dtype, device=device),
    #     'scale':   torch.ones((N, 3), dtype=dtype, device=device),
    #     'rotation': torch.cat([
    #         torch.ones((N, 1), dtype=dtype, device=device),
    #         torch.zeros((N, 3), dtype=dtype, device=device)
    #     ], dim=1),
    #     'rgb': flow_padded  # use flow vectors as "color"
    # }
 
  
 
    # gaussian_2d_curr = curr_res['mean_2d']
    # gaussian_2d_prev = prev_res['mean_2d']
    # flow_2d = gaussian_2d_curr - gaussian_2d_prev
    # # pad to 3 channels so it's RGB-like
    # flow_padded = torch.cat([flow_2d, torch.zeros_like(flow_2d[:, :1])], dim=1)

    # 2) Build a minimal Gaussian asset dict for the new renderer
    N = gaussian_3d_pos_curr.shape[0]
    device, dtype = gaussian_3d_pos_curr.device, gaussian_3d_pos_curr.dtype
    gauss_assets = {
        'mean_3d':   gaussian_3d_pos_curr,                             # world-space centers
        'opacity':   torch.ones((N,1), dtype=dtype, device=device),
        'scale':     torch.ones((N,3), dtype=dtype, device=device),
        'rotation':  torch.cat([                          # identity quaternion
                         torch.ones((N,1),dtype=dtype,device=device),
                         torch.zeros((N,3),dtype=dtype,device=device)
                      ], dim=1),
        'rgb':       flow_padded,                         # our “color” = flow vector
    }

    # 3) Render via the new GaussianRenderer
    gauss_renderer = GaussianRenderer()
    
    
    cam_param = cam_param[prev_time]  # must have been stored by your GUI
    flow_out = gauss_renderer(gauss_assets, render_resolution, cam_param, torch.zeros(3,device=device))
    rendered_flow_image  = flow_out['img']      # [3,H,W] flow map
    viewspace_pts  = flow_out['mean_2d']  # [N,3] new 2D means
 
 
  

    # Compute local scale change and loss using Gaussian deformations
    scales = self.gaussians._scaling
    rotations = self.gaussians._rotation
    # Get previous-timestep deformations
    idx_prev = self.prev_time_deform_T.index(prev_time)
    means3D_deform, scales_deform, rotations_deform, opacity_deform = (
        self.prev_means3D_deform_T[idx_prev],
        self.prev_scales_deform_T[idx_prev],
        self.prev_rotations_deform_T[idx_prev],
        self.prev_opacity_deform_T[idx_prev]
    )
    scales_final = self.gaussians.scaling_activation(scales_deform)
    rotations_final = self.gaussians.rotation_activation(rotations_deform)
    scale_change = scales_deform - scales
    local_scale_loss = scale_change[self.nn_indices] - scale_change.unsqueeze(1)
    local_scale_loss = torch.sqrt((local_scale_loss ** 2).sum(-1) * self.nn_weights + 1e-20).mean()

    # Compute local rigidity loss between current and previous Gaussians
    idx_curr = self.time_deform_T.index(curr_time)
    curr_rotations_deform = self.rotations_deform_T[idx_curr]
    curr_rotations_final = self.gaussians.rotation_activation(curr_rotations_deform)
    curr_rotations_final_inv = curr_rotations_final.clone()
    curr_rotations_final_inv[:, 1:] = -curr_rotations_final_inv[:, 1:]
    prev_nn_disp = gaussian_3d_pos_prev[self.nn_indices] - gaussian_3d_pos_prev.unsqueeze(1)
    curr_nn_disp = gaussian_3d_pos_curr[self.nn_indices] - gaussian_3d_pos_curr.unsqueeze(1)
    rel_rotmat = build_rotation(quat_mult(rotations_final, curr_rotations_final_inv))
    curr_nn_disp_warped = (rel_rotmat.transpose(2,1)[:, None] @ curr_nn_disp[:, :, :, None]).squeeze(-1)
    local_rigidity_loss = torch.sqrt(((prev_nn_disp - curr_nn_disp_warped) ** 2).sum(-1) * self.nn_weights + 1e-20).mean()

    return {
        "flow": rendered_flow_image,
        "viewspace_points": viewspace_pts,
        "scale_change": scale_change,
        "local_scale_loss": local_scale_loss,
        "gaussian_displacements": (gaussian_3d_pos_curr - gaussian_3d_pos_prev),
        "local_rigidity_loss": local_rigidity_loss,
    }


def spherical_to_cartesian(radius, elevation, azimuth, is_degree=True):
    if is_degree:
        elevation = np.radians(elevation)
        azimuth = np.radians(azimuth)

    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)

    return np.array([x, y, z])

def look_at(eye, target=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
    forward = (target - eye)
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    R = np.stack([right, up, forward], axis=1)
    t = -R.T @ eye

    return R, t

def pose_matrix_to_dict(pose_matrix,  device='cuda:0'):
    pose_tensor = torch.tensor(pose_matrix, dtype=torch.float32, device=device)
    R = pose_tensor[:3, :3]
    t = pose_tensor[:3, 3]
    # focal_tensor = torch.tensor(focal, dtype=torch.float32, device=device)
    # princpt_tensor = torch.tensor(princpt, dtype=torch.float32, device=device)

    focal = torch.tensor([572.5620, 578.3920], device=device)
    princpt = torch.tensor([638.5000, 359.0000], device=device)

    return {
        'R': R,
        't': t,
        'focal': focal,
        'princpt': princpt
    }


def random_camera_around_mean(center, dist_range, device='cuda:0'):
    """
    center: 평균 3D 좌표 (크기 3 벡터)
    dist_range: (min_dist, max_dist) 거리 범위 튜플
    device: 'cuda:0' 등
    """
    # 1) 평균 좌표 텐서화 (CPU) 및 장치 지정
    center = torch.tensor(center, dtype=torch.float32)

    # 2) 반구 방향 무작위 샘플링 (표준 정규분포 -> 정규화)
    v = torch.randn(3)
    v = v / torch.norm(v)
    # z 성분이 음수인 경우 반전하여 반구 제한
    if v[2] < 0:
        v = torch.tensor([v[0], v[1], -v[2]])

    # 3) 거리 무작위 샘플링
    d = torch.empty(1).uniform_(dist_range[0], dist_range[1]).item()
    cam_pos = center + v * d  # 카메라 위치 계산

    # 4) 카메라가 평균점을 바라보도록 회전 행렬 계산
    forward = center - cam_pos
    forward = forward / forward.norm()
    # 예를 들어 Z축을 up으로 사용
    up = torch.tensor([0.0, 0.0, 1.0])
    right = torch.cross(up, forward)
    right = right / right.norm()
    cam_up = torch.cross(forward, right)
    cam_up = cam_up / cam_up.norm()
    # 회전 행렬 구성 (열벡터로 x=right, y=cam_up, z=forward)
    R = torch.stack((right, cam_up, forward), dim=1)

    # 5) CUDA 텐서로 이동
    R = R.to(device)
    t = cam_pos.to(device)
    # 예시 초점 및 주점 (주어진 값 사용)
    focal = torch.tensor([572.5620, 578.3920], device=device)
    princpt = torch.tensor([638.5000, 359.0000], device=device)

    return {'R': R, 't': t, 'focal': focal, 'princpt': princpt}

def generate_camera_pose(elevation, azimuth, radius=1, is_degree=True, target=None):
    if target is None:
        target = np.array([0, 0, 0])

    eye = spherical_to_cartesian(radius, elevation, azimuth, is_degree)
    R, t = look_at(eye, target)

    pose = {
        'R': torch.tensor(R, dtype=torch.float32 , device='cuda:0'),
        't': torch.tensor(t, dtype=torch.float32 , device='cuda:0'),
        'focal': torch.tensor([572.5620, 578.3920], dtype=torch.float32, device='cuda:0'),
        'princpt': torch.tensor([638.5000, 359.0000], dtype=torch.float32, device='cuda:0'),
    }
 
    return pose



# def get_orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True, intrinsic=None, device='cuda:0'):
#     if is_degree:
#         elevation = np.deg2rad(elevation)
#         azimuth = np.deg2rad(azimuth)

#     x = radius * np.cos(elevation) * np.sin(azimuth)
#     y = - radius * np.sin(elevation)
#     z = radius * np.cos(elevation) * np.cos(azimuth)

#     if target is None:
#         target = np.zeros([3], dtype=np.float32)

#     campos = np.array([x, y, z]) + target

#     # look_at 함수 (기존의 look_at을 활용)
#     R_matrix = look_at(campos, target, opengl)

#     K = intrinsic[1]
    
#     fx  = K[0][0]
#     fy  = K[1][1]
#     cx  = K[0][2]
#     cy  = K[1][2]
        

#     focal  = torch.tensor([fx, fy], device=device)
#     princpt= torch.tensor([cx, cy], device=device)


#     pose_dict = {
#         'R': torch.tensor(R_matrix, dtype=torch.float32, device=device),
#         't': torch.tensor(campos, dtype=torch.float32, device=device),
#         'focal': focal.to(device),
#         'princpt': princpt.to(device)
#     }

#     return pose_dict


# def look_at(campos, target, opengl=True):
#     forward = safe_normalize(target - campos)
#     tmp = np.array([0, 1, 0], dtype=np.float32)

#     right = np.cross(tmp, forward)
#     right = safe_normalize(right)

#     up = np.cross(forward, right)
#     up = safe_normalize(up)

#     R = np.eye(3, dtype=np.float32)

#     if opengl:
#         R[0, :] = right
#         R[1, :] = up
#         R[2, :] = forward
#     else:
#         R[:, 0] = right
#         R[:, 1] = up
#         R[:, 2] = forward

#     return R


# def safe_normalize(v, eps=1e-8):
#     norm = np.linalg.norm(v)
#     return v / (norm + eps)



def get_intrinsics(self,intrinsic_path ):
        import os.path as osp
        
        intrinsics = {} 
        # intrinsic_path = getattr(self.args, "intrinsic_path", None)
        if intrinsic_path is None:
            # If not explicitly provided, assume cameras.txt is alongside cam_params.pkl
            intrinsic_path = osp.join(osp.dirname(self.args.cam_param_path), "cameras.txt")
        if os.path.exists(intrinsic_path):
            with open(intrinsic_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('#') or line.strip() == '':
                        continue  # skip comments/empty lines
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        cam_id = int(parts[0])
                        model = parts[1]
                        width = float(parts[2]); height = float(parts[3])
                        fx = float(parts[4]); fy = float(parts[5])
                        cx = float(parts[6]); cy = float(parts[7])
                        # Construct intrinsic matrix K for this camera
                        K = np.array([[fx, 0.0, cx],
                                    [0.0, fy, cy],
                                    [0.0, 0.0, 1.0]], dtype=np.float32)
                        intrinsics[cam_id] = K
        else:
            print(f"Warning: Intrinsic file not found at {intrinsic_path}.")
            
        self.intrinsics = intrinsics
        
        return intrinsics
    
    
@torch.no_grad()
def get_depth(self):
    
    depth_dir = self.opt.depth_dir
    if depth_dir is None or not os.path.isdir(depth_dir):
        raise ValueError("Depth directory is invalid or does not exist.")

    depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.npy')))
    if len(depth_files) == 0:
        raise RuntimeError(f'No *.npy depth files found in {depth_dir}')

    self.input_depth_list, self.input_depth_mask_list = [], []
    assert len(depth_files) == len(self.input_img_list), \
        "Depth-map 개수와 이미지 개수가 다릅니다."

    # 3) 깊이-맵 순회
    for i, depth_file in enumerate(depth_files):
        depth = np.load(depth_file)            # (H, W) or (H, W, 1)
        if depth.ndim == 2:                    # 채널 차원이 없으면 추가
            depth = depth[:, :, np.newaxis]

        # ----------- (기존 전처리 로직 유지) -----------
        H, W = depth.shape[:2]
        eroded_mask = scipy.ndimage.binary_erosion(
            self.input_mask_list[i].squeeze(-1) > 0.5,
            structure=np.ones((7, 7))
        )
        eroded_mask = (eroded_mask > 0.5)
        eroded_mask = eroded_mask[..., np.newaxis]

        median_depth = np.median(depth[eroded_mask])
        scaled_depth = (depth - median_depth) / \
                    np.abs(depth[eroded_mask] - median_depth).mean()
        masked_depth = scaled_depth * eroded_mask
        # ----------------------------------------------

        self.input_depth_list.append(masked_depth.astype(np.float32))
        # self.input_depth_mask_list.append(eroded_mask[:, :, np.newaxis]
        #                                 .astype(np.float32))
        self.input_depth_mask_list.append(eroded_mask.astype(np.float32))


    # 필요하다면 메모리 절약용 가비지 컬렉션
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()