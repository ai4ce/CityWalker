import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as TF
from decord import VideoReader, cpu
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import random

class CityWalkSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = self.create_indices()

    def create_indices(self):
        indices = []
        for start_idx, end_idx in self.dataset.video_ranges:
            video_indices = list(range(start_idx, end_idx))
            if self.dataset.mode == 'train':
                random.shuffle(video_indices)
            indices.extend(video_indices)
        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class CityWalkDataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.video_dir = cfg.data.video_dir
        self.pose_dir = cfg.data.pose_dir
        self.context_size = cfg.model.obs_encoder.context_size
        self.wp_length = cfg.model.decoder.len_traj_pred
        self.video_fps = cfg.data.video_fps
        self.pose_fps = cfg.data.pose_fps
        self.target_fps = cfg.data.target_fps
        self.num_workers = cfg.data.num_workers
        self.input_noise = cfg.data.input_noise
        self.search_window = cfg.data.search_window
        self.arrived_threshold = cfg.data.arrived_threshold

        # Load pose paths
        self.pose_path = [
            os.path.join(self.pose_dir, f)
            for f in sorted(os.listdir(self.pose_dir))
            if f.endswith('.txt')
        ]
        if mode == 'train':
            self.pose_path = self.pose_path[:cfg.data.num_train]
        elif mode == 'val':
            self.pose_path = self.pose_path[cfg.data.num_train:cfg.data.num_train + cfg.data.num_val]
        else:
            raise ValueError(f"Invalid mode {mode}")

        # Load corresponding video paths
        self.video_path = []
        for f in self.pose_path:
            video_file = os.path.basename(f).replace(".txt", ".mp4")
            video = os.path.join(self.video_dir, video_file)
            if not os.path.exists(video):
                raise FileNotFoundError(f"Video file {video} does not exist.")
            self.video_path.append(video)

        # Load poses and compute usable counts
        self.poses = []
        self.count = []
        for f in tqdm(self.pose_path, desc="Loading poses"):
            pose = np.loadtxt(f, delimiter=" ")[::max(1, self.pose_fps // self.target_fps), 1:]
            pose_nan = np.isnan(pose).any(axis=1)
            if np.any(pose_nan):
                first_nan_idx = np.argmin(pose_nan)
                pose = pose[:first_nan_idx]
            self.poses.append(pose)
            usable = pose.shape[0] - self.context_size - max(self.search_window, self.wp_length)
            self.count.append(max(usable, 0))  # Ensure non-negative

        # Remove pose files with zero usable samples
        valid_indices = [i for i, c in enumerate(self.count) if c > 0]
        self.poses = [self.poses[i] for i in valid_indices]
        self.video_path = [self.video_path[i] for i in valid_indices]
        self.count = [self.count[i] for i in valid_indices]

        # Build the look-up table and video_ranges
        self.lut = []
        self.video_ranges = []
        idx_counter = 0
        for video_idx, count in enumerate(self.count):
            start_idx = idx_counter
            for pose_start in range(count):
                self.lut.append((video_idx, pose_start))
                idx_counter += 1
            end_idx = idx_counter
            self.video_ranges.append((start_idx, end_idx))
        assert len(self.lut) > 0, "No usable samples found."

        # Initialize the video reader cache per worker
        self.video_reader_cache = {'video_idx': None, 'video_reader': None}

    def __len__(self):
        return len(self.lut)

    def __getitem__(self, index):
        video_idx, pose_start = self.lut[index]

        # Retrieve or create the VideoReader for the current video
        if self.video_reader_cache['video_idx'] != video_idx:
            # Replace the old VideoReader with the new one
            self.video_reader_cache['video_reader'] = VideoReader(self.video_path[video_idx], ctx=cpu(0))
            self.video_reader_cache['video_idx'] = video_idx
        video_reader = self.video_reader_cache['video_reader']

        frame_multiplier = self.video_fps // self.target_fps
        start_frame_idx = pose_start * frame_multiplier
        frame_indices = start_frame_idx + np.arange(self.context_size) * frame_multiplier

        # Ensure frame indices are within the video length
        num_frames = len(video_reader)
        frame_indices = [min(idx, num_frames - 1) for idx in frame_indices]

        # Load the required frames
        frames = video_reader.get_batch(frame_indices).asnumpy()

        # Process frames
        frames = self.process_frames(frames)

        # Get pose data
        pose = self.poses[video_idx]

        # Get input and future poses
        input_poses, future_poses = self.get_input_and_future_poses(pose, pose_start)
        original_input_poses = np.copy(input_poses)  # Store original poses before noise

        # Select target pose
        target_pose = self.select_target_pose(future_poses)

        # Determine arrived label
        arrived = self.determine_arrived_label(input_poses[-1, :3], target_pose[:3])

        # Extract waypoints
        waypoint_poses = self.extract_waypoints(pose, pose_start)

        # Add noise if necessary
        if self.input_noise > 0:
            input_poses = self.add_noise(input_poses)

        # Transform poses
        current_pose = input_poses[-1]
        transformed_input_positions = self.transform_poses(input_poses, current_pose)
        transformed_original_input_positions = self.transform_poses(original_input_poses, current_pose)
        waypoints_transformed = self.transform_waypoints(waypoint_poses, current_pose)
        target_transformed = self.transform_target_pose(target_pose, current_pose)

        # Convert data to tensors
        input_positions = torch.tensor(transformed_input_positions[:, [0, 2]], dtype=torch.float32)
        waypoints_transformed = torch.tensor(waypoints_transformed[:, [0, 2]], dtype=torch.float32)
        arrived = torch.tensor(arrived, dtype=torch.float32)
        sample = {
            'video_frames': frames,
            'input_positions': input_positions,
            'waypoints': waypoints_transformed,
            'arrived': arrived
        }

        # For visualization during validation
        if self.mode == 'val':
            original_input_positions = torch.tensor(transformed_original_input_positions[:, [0, 2]], dtype=torch.float32)
            noisy_input_positions = torch.tensor(transformed_input_positions[:, [0, 2]], dtype=torch.float32)
            target_transformed_position = torch.tensor(target_transformed[[0, 2]], dtype=torch.float32)  # Only X and Z
            sample['original_input_positions'] = original_input_positions
            sample['noisy_input_positions'] = noisy_input_positions
            sample['gt_waypoints'] = waypoints_transformed
            sample['target_transformed'] = target_transformed_position  # Add target coordinate
        return sample

    def transform_poses(self, poses, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrices = self.poses_to_matrices(poses)
        transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        positions = transformed_matrices[:, :3, 3]
        return positions

    def get_input_and_future_poses(self, pose, pose_start):
        input_poses = pose[pose_start: pose_start + self.context_size]
        search_end = min(pose_start + self.context_size + self.search_window, pose.shape[0])
        future_poses = pose[pose_start + self.context_size: search_end]
        if future_poses.shape[0] == 0:
            raise IndexError(f"No future poses available for index {pose_start}.")
        return input_poses, future_poses

    def select_target_pose(self, future_poses):
        target_idx = random.randint(0, future_poses.shape[0] - 1)
        target_pose = future_poses[target_idx]
        return target_pose

    def determine_arrived_label(self, current_pos, target_pos):
        distance_to_goal = np.linalg.norm(target_pos - current_pos, axis=0)
        arrived = distance_to_goal <= self.arrived_threshold
        return arrived

    def extract_waypoints(self, pose, pose_start):
        waypoint_start = pose_start + self.context_size
        waypoint_end = waypoint_start + self.wp_length
        waypoint_poses = pose[waypoint_start: waypoint_end]
        return waypoint_poses

    def add_noise(self, input_poses):
        noise = np.random.normal(0, self.input_noise, input_poses[:, :3].shape)
        input_poses[:, :3] += noise
        return input_poses

    def process_frames(self, frames):
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0  # Corrected normalization
        # frames = TF.resize(frames, [224, 224])
        # frames = TF.normalize(frames, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return frames

    def transform_waypoints(self, waypoint_poses, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        waypoint_matrices = self.poses_to_matrices(waypoint_poses)
        transformed_waypoint_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], waypoint_matrices)
        waypoints_positions = transformed_waypoint_matrices[:, :3, 3]
        return waypoints_positions

    def transform_target_pose(self, target_pose, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        target_pose_matrix = self.pose_to_matrix(target_pose)
        transformed_target_matrix = np.matmul(current_pose_inv, target_pose_matrix)
        target_position = transformed_target_matrix[:3, 3]
        return target_position

    def pose_to_matrix(self, pose):
        position = pose[:3]
        rotation = R.from_quat(pose[3:])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = position
        return matrix

    def poses_to_matrices(self, poses):
        positions = poses[:, :3]
        quats = poses[:, 3:]
        rotations = R.from_quat(quats)
        matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
        matrices[:, :3, :3] = rotations.as_matrix()
        matrices[:, :3, 3] = positions
        return matrices
