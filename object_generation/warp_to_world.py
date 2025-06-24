from scipy.spatial.transform import Rotation 
import cv2
import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import copy
import glob
from PIL import Image
from torchmetrics import PearsonCorrCoef

from gaussian_model_4d import GaussianModel
from utils.general_utils import get_expon_lr_func
from nets.module import GaussianRenderer

class GlobalOptimizer(nn.Module):
    def __init__(self, args):
        """
        Initialize the GlobalOptimizer class.
        
        Args:
            args: Arguments containing configuration settings
        """
        super(GlobalOptimizer, self).__init__()
        self.args = args
        self.device = torch.device("cuda")
        self.pearson = PearsonCorrCoef().to(self.device, non_blocking=True)
        
        self.depth_map_dict = {}
        
        # Initialize the gaussian renderer
        self.gaussian_renderer = GaussianRenderer()
        
        # Load input data
        self.load_input_data(args)
        
        # Initialize canonical gaussians
        self.initialize_canonical_gaussians(args)
        
        # Initialize global motion parameters
        self.initialize_global_motion()
    
    def load_input_data(self, args):
        """
        Load and prepare input data including images, masks, and camera parameters.
        
        Args:
            args: Arguments containing paths to input data
        """
        # Load images and masks
        self.load_images_and_masks(args.img_dir, args.mask_dir)
        
        # Load camera parameters
        self.load_camera_parameters(args.cam_param_path)
        
        # Calculate object parameters (center and scale)
        self.calculate_object_parameters()
    
    def load_images_and_masks(self, img_dir, mask_dir):
        """
        Load images and masks from directories.
        
        Args:
            img_dir: Directory containing images
            mask_dir: Directory containing masks
        """
        # Get file lists
        img_files = sorted(glob.glob(os.path.join(img_dir, '*')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*')))
        
        assert len(img_files) == len(mask_files), "Number of images and masks must match"
        
        self.vid_length = len(img_files)
        
        self.input_img_list = []
        self.input_mask_list = []
        
        # Load images and masks
        for img_file, mask_file in zip(img_files, mask_files):
            # Load image
            img = Image.open(img_file)
            img = np.array(img)[:, :, :3].astype(np.float32) / 255.0
            self.input_img_list.append(img)
            
            # Load mask
            mask = Image.open(mask_file)
            mask = np.array(mask).astype(np.float32) / 255.0
            if len(mask.shape) == 3:
                mask = mask[:, :, 0:1]
            else:
                mask = mask[:, :, np.newaxis]
            self.input_mask_list.append(mask)
        
        # Convert to PyTorch tensors
        self.input_img_torch_list = []
        self.input_mask_torch_list = []
        
        for img, mask in zip(self.input_img_list, self.input_mask_list):
            self.input_img_torch_list.append(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device))
            self.input_mask_torch_list.append(torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).to(self.device))
    
    def load_camera_parameters(self, cam_param_path):
        """
        Load camera parameters from file.
        
        Args:
            cam_param_path: Path to camera parameters file
        """
        with open(cam_param_path, 'rb') as f:
            self.cam_params = pickle.load(f)
    
    def calculate_object_parameters(self):
        """ 
        Calculate object center and scale from masks.
        """
        self.obj_cx_list = []
        self.obj_cy_list = []
        self.obj_scale_list = []
         
        # Calculate focal scales for 3D to 2D conversion
        if 'focal' in self.cam_params:
            focal = self.cam_params['focal'][0]
            self.x_scale = 1.0 / focal[0].item()
            self.y_scale = 1.0 / focal[1].item()
        else:
            self.x_scale = 1.0
            self.y_scale = 1.0
        
        # Load initial scale if available
        initial_scale_path = os.path.join(self.args.outdir, "gaussians", f"{self.args.save_path}_global_motion.pkl")
        if os.path.exists(initial_scale_path):
            with open(initial_scale_path, 'rb') as f:
                input_scale0 = pickle.load(f)['scale'].squeeze()[0]
            self.initial_scale = input_scale0
        else:
            self.initial_scale = 1.0
        
        for mask_torch in self.input_mask_torch_list:
            N, C, H, W = mask_torch.shape
            mask = mask_torch > 0.5
            nonzero_idxes = torch.nonzero(mask[0, 0])
            
            if len(nonzero_idxes) > 0:
                # Find bbox
                min_x = nonzero_idxes[:, 1].min()
                max_x = nonzero_idxes[:, 1].max()
                min_y = nonzero_idxes[:, 0].min()
                max_y = nonzero_idxes[:, 0].max()
                
                # Find cx cy
                cx = (max_x + min_x) / 2
                cx = ((cx / W) * 2 - 1)
                cy = (max_y + min_y) / 2
                cy = ((cy / H) * 2 - 1)
                
                # Find maximum possible scale
                width = (max_x - min_x) / W
                height = (max_y - min_y) / H
                scale_x = width / 0.975
                scale_y = height / 0.975
                max_scale = max(scale_x, scale_y)
                
                # If the scale from the first frame doesn't clip the object, then stick with it
                scale = max(max_scale, self.initial_scale)
                
                self.obj_cx_list.append(cx)
                self.obj_cy_list.append(cy)
                self.obj_scale_list.append(scale)
            else:
                # If mask is empty, use default values
                self.obj_cx_list.append(0.0)
                self.obj_cy_list.append(0.0)
                self.obj_scale_list.append(1.0)
    
    def initialize_canonical_gaussians(self, args):
        """
        Initialize canonical gaussians from files or from scratch.
        Args:
            args: Arguments containing paths to canonical gaussian files
        """ 
        self.canonical_gaussians = []
         
        # If canonical dir is provided, load canonical gaussians from files
        if hasattr(args, 'canonical_dir') and args.canonical_dir is not None:
            for t in range(self.vid_length):
                canon_path = os.path.join(args.canonical_dir, f'canonical_{t}.pkl')
                if os.path.exists(canon_path):
                    with open(canon_path, 'rb') as f:
                        self.canonical_gaussians.append(pickle.load(f))
                        
                else:
                    print(f"Warning: Canonical gaussian file not found for time step {t}")
                    self.canonical_gaussians.append(None)
        
        # Otherwise, load 4D gaussian model and use it to generate canonical gaussians
        else:
            # Initialize 4D gaussian model
            self.gaussians = GaussianModel(args.sh_degree, args)
            
            # Load model from file if provided
            if args.load is not None:
                self.gaussians.load_ply(args.load)
                if hasattr(args, 'deformation_path') and args.deformation_path is not None:
                    self.gaussians.load_model(os.path.dirname(args.deformation_path), os.path.basename(args.deformation_path))
            
            # Generate canonical gaussians for each time step
            for t in range(self.vid_length):
                # Get deformed gaussians at time t
                time = torch.tensor(t).to(self.device).repeat(self.gaussians.get_xyz.shape[0], 1)
                time = ((time.float() / self.vid_length) - 0.5) * 2
                means3D, scales, rotations, opacity = self.gaussians._deformation(
                    self.gaussians.get_xyz,
                    self.gaussians._scaling,
                    self.gaussians._rotation,
                    self.gaussians._opacity,
                    time
                )
                
                # Create asset dict
                asset = {
                    'mean_3d': means3D,
                    'scale': self.gaussians.scaling_activation(scales),
                    'rotation': self.gaussians.rotation_activation(rotations),
                    'opacity': self.gaussians.opacity_activation(opacity),
                    'rgb': self.gaussians.get_features
                }
                
                self.canonical_gaussians.append(asset)
    
    
    
    def initialize_global_motion(self):
        """
        Initialize the global motion parameters (per-frame translation and scale)
        using mask centroid, depth, and camera calibration data.
        """
         
        # Initialize translation and scale parameters for all frames
        self.translations = nn.Parameter(
            torch.zeros((self.vid_length, 3), dtype=torch.float32, device=self.device),
            requires_grad=False
        )

        self.z_translations = nn.Parameter(
            torch.zeros((self.vid_length, 1), dtype=torch.float32, device=self.device),
            requires_grad=True
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
        intrinsic_path = getattr(self.args, "intrinsic_path", None)
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
        depth_dir = getattr(self.args, "depth_dir", None)
        depth_files = []
        if depth_dir is not None and os.path.isdir(depth_dir):
            depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.npy')))
            if len(depth_files) != self.vid_length:
                print("Warning: Number of depth files does not match number of frames.")
        # If depth_dir is not provided, we will derive depth path per frame from mask filename
        
        # Iterate over each frame to compute translation and set scale
        for idx in range(self.vid_length):
            # Check if the object is present in this frame (mask has any foreground)
            if torch.sum(self.input_mask_torch_list[idx]) == 0:
                # No object in frame: keep default translation [0,0,0] and scale 1.0
                self.translations.data[idx] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
                self.scales.data[idx] = 1.0
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
                
                self.depth_map_dict[idx] = depth_map
                
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
                    cam_extrinsic = self.to_cpu(cam_extrinsic)
                    
                elif isinstance(self.cam_params, (list, tuple)) and idx < len(self.cam_params):
                    cam_extrinsic = self.cam_params[idx] 
                    cam_extrinsic = self.to_cpu(cam_extrinsic)
                    
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
            
            
            # Create optimizer for global motion parameters
            # self.optimizer = torch.optim.AdamW([
            #     {'params': [self.translations], 'lr': self.args.position_lr_init, 'weight_decay': 0., 'name': 'translation'},
            #     {'params': [self.scales], 'lr': self.args.position_lr_init * 0.1, 'weight_decay': 0., 'name': 'scale'}
            # ], lr=0.0, eps=1e-15)
            

            self.optimizer = torch.optim.AdamW([
                {'params': [self.translations], 'lr': self.args.position_lr_init, 'weight_decay': 0., 'name': 'translation'},
                {'params': [self.scales],      'lr': self.args.position_lr_init * 0.1, 'weight_decay': 0., 'name': 'scale'},
                {'params': [self.rotations],   'lr': self.args.position_lr_init, 'weight_decay': 0., 'name': 'rotation'},
            ], lr=0.0, eps=1e-15)



            self.scheduler_args = get_expon_lr_func(
                lr_init=self.args.position_lr_init,
                lr_final=self.args.position_lr_final,
                lr_delay_mult=self.args.position_lr_delay_mult,
                max_steps=self.args.position_lr_max_steps
            )
        
            self.step = 0
             
        
        self.save_global_transformed_gaussians(canonical_asset=self.canonical_gaussians
                                                ,output_dir= self.args.outdir)
        
        exit()
        
            

    def quaternion_rotate(self, points, quat):
        """
        Rotate Nx3 tensor of points by a quaternion quat (x,y,z,w).
        """
        x, y, z, w = quat  # assuming quat is tensor [x,y,z,w]
        # Pre-compute products
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        # Construct rotation matrix components
        rot_mat = torch.tensor([
            [1 - 2*(yy + zz),    2*(xy - wz),    2*(xz + wy)],
            [2*(xy + wz),    1 - 2*(xx + zz),    2*(yz - wx)],
            [2*(xz - wy),        2*(yz + wx), 1 - 2*(xx + yy)]
        ], device=points.device, dtype=points.dtype)
        # Apply rotation
        return points @ rot_mat.T

    def to_cpu(self, obj):
        """
        모든 torch.Tensor(또는 텐서를 포함한 컨테이너)를 재귀적으로 순회하며
        .cpu()를 호출한 새 객체를 반환합니다.
        - dict, list, tuple, set 등을 안전하게 처리합니다.
        - 텐서가 아닌 값은 그대로 반환합니다.
        """
        if isinstance(obj, torch.Tensor):
            return obj.cpu()

        # 파이썬 내장 컨테이너 처리
        if isinstance(obj, dict):
            return {k: self.to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.to_cpu(v) for v in obj)
        if isinstance(obj, set):
            return {self.to_cpu(v) for v in obj}

        # 기타 사용자 정의 클래스(예: dataclass)나 네임드튜플 등 처리
        if hasattr(obj, "__dict__"):
            # __dict__를 가진 객체면, 얕은 복사 후 속성별로 재귀 적용
            new_obj = obj.__class__.__new__(obj.__class__)
            for k, v in obj.__dict__.items():
                setattr(new_obj, k, self.to_cpu(v))
            return new_obj

        # 위 조건에 걸리지 않으면 원본 그대로
        return obj
    def update_learning_rate(self, step):
        """
        Update learning rate based on the current step.
        
        Args:
            step: Current optimization step
        """
        for param_group in self.optimizer.param_groups:
            lr = self.scheduler_args(step)
            param_group['lr'] = lr * (0.1 if param_group['name'] == 'scale' else 1.0)
    

    def save_global_transformed_gaussians(self, canonical_asset, output_dir):
        """
        Save globally transformed Gaussian parameters for all time indices to output directory.

        Args:
            assets (list of dict): List of canonical Gaussian parameters for each time index.
            output_dir (str): Directory to save Gaussian files.
        """
        os.makedirs(output_dir, exist_ok=True)

        for time_idx, asset in enumerate(canonical_asset):
            
            transformed_asset = self.apply_global_motion(asset, time_idx)

            # Convert tensors to numpy arrays for saving
            asset_to_save = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                            for key, value in transformed_asset.items()}
            

            # Save the transformed Gaussian
            output_file = os.path.join(output_dir, f"global_gaussian_{time_idx}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(asset_to_save, f)

        print(f"Saved globally transformed Gaussians to {output_dir}")

    def mask_distance_loss(self, mask1, mask2, p=1):
        """
        두 바이너리 마스크의 distance transform 맵 사이의 Lp distance loss (p=1: L1, p=2: L2)
        mask1, mask2: (B, 1, H, W) torch tensor, 0/1 binary
        """
        # numpy 변환 후 distance transform 계산
        dt1 = torch.from_numpy(self.distance_transform(mask1.cpu().numpy())).to(mask1.device)
        dt2 = torch.from_numpy(self.distance_transform(mask2.cpu().numpy())).to(mask2.device)
        
        if p == 1:
            return torch.abs(dt1 - dt2).mean()
        elif p == 2:
            return ((dt1 - dt2) ** 2).mean()
        else:
            raise ValueError("p should be 1 or 2")
    def balanced_mask_loss(self, pred, target, mask):
        """
        Compute a balanced loss for masked and non-masked regions.
        
        Args:
            pred: Predicted image
            target: Target image
            mask: Binary mask
        
        Returns:
            Balanced loss value
        """
        # Compute loss over mask and non-mask regions separately
        if mask.sum() > 0:
            masked_loss = (F.mse_loss(pred, target, reduction='none') * mask).sum() / mask.sum()
        else:
            masked_loss = 0.
        
        if (1 - mask).sum() > 0:
            masked_loss_empty = (F.mse_loss(pred, target, reduction='none') * (1 - mask)).sum() / (1 - mask).sum()
        else:
            masked_loss_empty = 0.
        
        return masked_loss + masked_loss_empty
    
    
    
    def balanced_mask_loss_2(self, pred, target, mask):
        """
        Compute a balanced loss for masked and non-masked regions.

        Args:
            pred:   torch.Tensor of shape [B, C, H, W]
            target: torch.Tensor or np.ndarray, same shape as pred
            mask:   torch.Tensor or np.ndarray, binary mask of shape [B, H, W] or [B, 1, H, W]

        Returns:
            torch.Tensor scalar: balanced mask loss
        """
        # 1) NumPy → Tensor 변환 & device/dtype 통일
        if not torch.is_tensor(target):
            target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, dtype=pred.dtype, device=pred.device)
        target = target.to(pred.dtype).to(pred.device)
        mask   = mask.to(pred.dtype).to(pred.device)

        # 2) mask shape 맞추기 ([B, H, W] → [B, 1, H, W])
        if mask.dim() == pred.dim() - 1:
            mask = mask.unsqueeze(1)

        # 3) MSE map 계산
        loss_map = F.mse_loss(pred, target, reduction='none')  # [B, C, H, W]

        # 4) mask, inverse-mask 합 계산
        mask_sum = float(mask.sum().item())
        inv_mask = 1.0 - mask
        inv_sum  = float(inv_mask.sum().item())

        # 5) 영역별 평균 손실
        if mask_sum > 0:
            masked_loss       = (loss_map * mask).sum() / mask_sum
        else:
            masked_loss = pred.new_tensor(0.)

        if inv_sum > 0:
            masked_loss_empty = (loss_map * inv_mask).sum() / inv_sum
        else:
            masked_loss_empty = pred.new_tensor(0.)

        return masked_loss + masked_loss_empty

    def apply_global_motion(self, asset, time_idx):
        transformed_asset = {}
        mean_3d = asset['mean_3d']
        if isinstance(mean_3d, np.ndarray):
            mean_3d = torch.from_numpy(mean_3d).to(self.device)
        scale = asset['scale']
        if isinstance(scale, np.ndarray):
            scale = torch.from_numpy(scale).to(self.device)
        opacity = asset['opacity']
        if isinstance(opacity, np.ndarray):
            opacity = torch.from_numpy(opacity).to(self.device)
        rgb = asset['rgb']
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).to(self.device)


        axis_flags = (False, True, True)
        for axis, flag in enumerate(axis_flags):
            if flag:
                mean_3d[:, axis] *= -1


        # Fetch the learned parameters for this frame
        t = self.translations[time_idx]  # (3,)
        s = self.scales[time_idx]        # scalar
        q = self.rotations[time_idx]     # (4,)

        t_z = self.z_translations[time_idx]

        # Apply scale, then rotate, then translate
        points = mean_3d * s  # scale
        rotated = self.quaternion_rotate(points, q)  # rotate by quaternion
        transformed_asset['mean_3d'] =  t + rotated  # translate


        # print(transformed_asset['mean_3d'].shape)
        # print(transformed_asset['mean_3d'][:,2].shape)

        transformed_asset['mean_3d'][:,2] += t_z

        # Update other attributes similarly
        transformed_asset['scale'] = scale * s

        # We keep the original asset rotation or identity; 
        # if needed we could also compose rotations here.
        # transformed_asset['rotation'] = asset.get('rotation', None)  

        rotation = asset['rotation']
        if isinstance(rotation, np.ndarray):
            rotation = torch.from_numpy(rotation).to(self.device)

        transformed_asset['rotation'] = rotation
        
        transformed_asset['opacity'] = opacity
        transformed_asset['rgb'] = rgb

        return transformed_asset
    
    # def apply_global_motion(self, asset, time_idx):
    #     transformed_asset = {}
    #     # Convert to tensor if not already
    #     mean_3d = asset['mean_3d']
    #     if isinstance(mean_3d, np.ndarray):
    #         mean_3d = torch.from_numpy(mean_3d).to(self.device)
    #     scale = asset['scale']
    #     if isinstance(scale, np.ndarray):
    #         scale = torch.from_numpy(scale).to(self.device)
    #     rotation = asset['rotation']
    #     if isinstance(rotation, np.ndarray):
    #         rotation = torch.from_numpy(rotation).to(self.device)
    #     opacity = asset['opacity']
    #     if isinstance(opacity, np.ndarray):
    #         opacity = torch.from_numpy(opacity).to(self.device)
        
    #     rgb = asset['rgb']
    #     if isinstance(rgb, np.ndarray):
    #         rgb = torch.from_numpy(rgb).to(self.device)

    #     transformed_asset['mean_3d'] = mean_3d * self.scales[time_idx] + self.translations[time_idx]
    #     transformed_asset['scale'] = scale * self.scales[time_idx]
    #     transformed_asset['rotation'] = rotation
    #     transformed_asset['opacity'] = opacity
    #     transformed_asset['rgb'] = rgb
    #     return transformed_asset
    
    def optimize_global_motion(self, iterations):
        """
        Optimize the global motion parameters.
        
        Args:
            iterations: Number of optimization iterations
        """
        for _ in range(iterations):
            self.step += 1
            step_ratio = min(1, self.step / self.args.iters)
            
            # Update learning rate
            self.update_learning_rate(self.step)
            
            loss = 0
            rand_timesteps = np.random.choice(np.arange(self.vid_length), self.args.batch_size, replace=False).tolist()

            
            
            # if (self.step) % 10 == 0:
            #     path = f'./trained_depth_3tz_2/{self.step}/'
            #     os.makedirs(path,  exist_ok=True)
            #     self.save_global_transformed_gaussians(canonical_asset=self.canonical_gaussians
            #                                     ,output_dir= path)
                
            # for time_idx in range(64,181):
            # # for time_idx in range(284):
            for time_idx in rand_timesteps:
                
                img_height, img_width = self.input_img_torch_list[time_idx].shape[-2:]
                
                # Get canonical gaussian asset
                canonical_asset = self.canonical_gaussians[time_idx]
                # Apply global motion
                transformed_asset = self.apply_global_motion(canonical_asset, time_idx)
                
      
                # Render with global motion
                render_result = self.gaussian_renderer(
                    transformed_asset,
                    (img_height, img_width),
                    self.cam_params[time_idx]
                )
                
                # Get target image and mask
                target_img = self.input_img_torch_list[time_idx][0]
                target_mask = (self.input_mask_torch_list[time_idx] > 0.5).float()[0]
                
                
                # RGB loss 
                image = render_result["img"]
                if target_mask.sum() > 0:
                    loss = loss + 10 * step_ratio * self.balanced_mask_loss(image, target_img, target_mask)
                    # loss = loss + 1000 * step_ratio * self.balanced_mask_loss(image, target_img, target_mask)
                
                # Mask loss
                mask = render_result["mask"]
                if target_mask.sum() > 0:
                    loss = loss + 10 * step_ratio * self.balanced_mask_loss(mask, target_mask, target_mask)
                    # loss = loss + 10000 * step_ratio * self.balanced_mask_loss(mask, target_mask, target_mask)
                    
                    
                
                mask_bool = mask > 0.5
                #    또는, 만약 mask 값이 0~1 사이 실수라면
                # mask_bool = (mask > 0.5)

                # 2) 이제 안전하게 인덱싱
                
                depthmap = render_result["depthmap"]      # 예: [B, H, W] 혹은 [H, W]
                depth = depthmap[mask_bool].mean()
                
                
                # print("depth.shape",depth.shape)
                
                if target_mask.sum() > 0:
                    target_depth = self.depth_map_dict[time_idx]

                    target_depth = torch.tensor(target_depth, device='cuda:0').float()

                    # target_depth = target_depth[target_mask]
                    target_mask = target_mask.squeeze() > 0.5
                    depth_vals = target_depth[target_mask].mean()

                    loss = loss + 10 * step_ratio * F.l1_loss(depth, depth_vals, reduction='none') 


                    # print(f"depth_pred: {depth} , GT depth: {depth_vals}" )

                    
                    
                    # loss = loss + 10000 * step_ratio * self.balanced_mask_loss_2(depth, target_depth, target_mask)
                     
                    
                # loss = L1( depth.mean() - depth_map[target_mask])
                # New loss
                # mask = render_result["mask"]
                # if target_mask.sum() > 0:
                #     # loss = loss + 100 * step_ratio * self.balanced_mask_loss(mask, target_mask, target_mask)
                #     loss = loss + 100000 * step_ratio * self.mask_distance_loss(target_mask, target_mask, p=1)
                    
                
                # vis = True
                # root_vis_dir = osp.join('./vis/' + args.name, 'train_vis')
                # # root_mask_dir = osp.join('./vis', 'train_mask_vis')
                
                
                
                # vis_dir  = osp.join(root_vis_dir, str(time_idx))
                # # mask_dir = osp.join(root_mask_dir, str(time_idx))
                # os.makedirs(vis_dir,  exist_ok=True)
                # # os.makedirs(mask_dir, exist_ok=True)
            
                
                # if vis:
                #     object_img_np = image.detach().cpu().numpy().transpose(1, 2, 0)
                #     object_img_np = np.clip(object_img_np * 255, 0, 255).astype(np.uint8)
                #     object_img_bgr = object_img_np[..., ::-1]
                    
                #     target_img_np = target_img.detach().cpu().numpy().transpose(1, 2, 0)
                #     target_img_np = np.clip(target_img_np * 255, 0, 255).astype(np.uint8)
                #     target_img_bgr = target_img_np[..., ::-1]
                    
                    
                #     mask_np = mask.squeeze(0).detach().cpu().numpy() 
                #     mask_np_uint8 = (mask_np * 255).clip(0, 255).astype(np.uint8)
                    
                #     target_mask_np = target_mask.squeeze(0).detach().cpu().numpy() 
                #     target_mask_np_uint8 = (target_mask_np * 255).clip(0, 255).astype(np.uint8)


                #     cv2.imwrite(osp.join(vis_dir, f'{self.step}_train.png'), object_img_bgr)
                #     cv2.imwrite(osp.join(vis_dir, f'{self.step}_target.png'), target_img_bgr)
                    
                    # cv2.imwrite(osp.join(mask_dir, f'{self.step}_train.png'), mask_np_uint8)
                    # cv2.imwrite(osp.join(mask_dir, f'{self.step}_target.png'), target_mask_np_uint8)
                    
            print(f"{self.step}/ loss: ",loss.item())
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            
    
    def save_global_motion(self, output_path):
        """
        Save the optimized global motion parameters to a file.
        
        Args:
            output_path: Path to save the global motion parameters
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        params_dict = {
            'translation': self.translations.detach(),
            'scale': self.scales.detach(),
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(params_dict, f)
        
        print(f"Global motion parameters saved to {output_path}")
    
    def train(self, iterations):
        """
        Train the global optimizer.
        
        Args:
            iterations: Number of training iterations
        """
        print("Starting global optimization...")
        self.optimize_global_motion(iterations)
        
        output_path = os.path.join(self.args.outdir, "gaussians", f"{self.args.save_path}_global_motion.pkl")
        self.save_global_motion(output_path)
        
        # self.save_global_transformed_gaussians(canonical_asset=self.canonical_gaussians
        #                                         ,output_dir= './trained_gaussians')
        
        print("Global optimization completed.")




if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # 1) Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True,
        help="path to the YAML configuration file"
    )
    parser.add_argument(
        "--img_dir", required=True,
        help="directory containing input images"
    )
    parser.add_argument(
        "--mask_dir", required=True,
        help="directory containing object masks"
    )
    parser.add_argument(
        "--cam_param_path", required=True,
        help="path to COLMAP cameras.txt (or .pkl)"
    )
    parser.add_argument(
        "--depth_dir", required=True,
        help="directory containing depth maps"
    )
    parser.add_argument(
        "--canonical_dir", required=True,
        help="directory with saved canonical Gaussians"
    )
    parser.add_argument(
        "--intrinsic_path", required=True,
        help="path to the intrinsics file"
    )
    parser.add_argument(
        "--outdir", required=True,
        help="output directory for world-warped results"
    )
    parser.add_argument(
        "--save_prefix", default="sample",
        help="prefix for saved output files"
    )
    args, extras = parser.parse_known_args()

    # 2) Merge CLI args into OmegaConf
    opt = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_cli(extras),
        OmegaConf.create({
            "img_dir":        args.img_dir,
            "mask_dir":       args.mask_dir,
            "cam_param_path": args.cam_param_path,
            "depth_dir":      args.depth_dir,
            "canonical_dir":  args.canonical_dir,
            "intrinsic_path": args.intrinsic_path,
            "outdir":         args.outdir,
            "save_prefix":    args.save_prefix,
        })
    )

    # Create optimizer
    optimizer = GlobalOptimizer(opt)
    
    # Train
    optimizer.train(opt.iters)

