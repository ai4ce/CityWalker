import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import random
from PIL import Image

class UrbanNavDataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.pose_dir = cfg.data.pose_dir
        self.image_root_dir = cfg.data.image_root_dir
        self.context_size = cfg.model.obs_encoder.context_size
        self.wp_length = cfg.model.decoder.len_traj_pred
        self.target_fps = cfg.data.target_fps
        self.num_workers = cfg.data.num_workers
        self.search_window = cfg.data.search_window
        self.arrived_threshold = cfg.data.arrived_threshold
        self.arrived_prob = cfg.data.arrived_prob

        # Load pose paths
        # self.pose_path = [
        #     os.path.join(self.pose_dir, f)
        #     for f in sorted(os.listdir(self.pose_dir))
        #     if f.startswith('match_gps_pose') and f.endswith('.txt')
        # ]
        pose_files = [
            "match_gps_pose6.txt",
            "match_gps_pose7.txt",
            "match_gps_pose8.txt",
            "match_gps_pose9.txt",
            "match_gps_pose11.txt",
        ]
        self.pose_path = [os.path.join(self.pose_dir, f) for f in pose_files]

        if mode == 'train':
            self.pose_path = self.pose_path[:cfg.data.num_train]
        elif mode == 'val':
            self.pose_path = self.pose_path[cfg.data.num_train: cfg.data.num_train + cfg.data.num_val]
        elif mode == 'test':
            self.pose_path = self.pose_path[cfg.data.num_train + cfg.data.num_val: cfg.data.num_train + cfg.data.num_val + cfg.data.num_test]
        else:
            raise ValueError(f"Invalid mode {mode}")

        # Initialize storage
        self.gps_positions = []
        self.poses = []
        self.image_names = []
        self.count = []
        self.image_folders = []

        for f in tqdm(self.pose_path, desc="Loading data"):
            seq_idx = ''.join(filter(str.isdigit, os.path.basename(f)))
            image_folder = os.path.join(self.image_root_dir, f'dog_nav_{seq_idx}')
            if not os.path.exists(image_folder):
                raise FileNotFoundError(f"Image folder {image_folder} does not exist.")
            self.image_folders.append(image_folder)

            with open(f, 'r') as file:
                lines = file.readlines()

            gps_positions = []
            poses = []
            images = []

            # Reference GPS point for local coordinate conversion
            ref_lat, ref_lon, ref_alt = None, None, None

            for i in range(0, len(lines), 2):
                gps_line = lines[i].strip()
                pose_line = lines[i+1].strip()

                # Parse GPS data
                gps_tokens = gps_line.split(',')
                if len(gps_tokens) < 8:
                    continue
                # timestamp = float(gps_tokens[0])
                latitude = float(gps_tokens[1])
                longitude = float(gps_tokens[2])
                # accuracy = float(gps_tokens[3])
                altitude = float(gps_tokens[4])
                # altitudeAccuracy = float(gps_tokens[5])
                # heading = gps_tokens[6]
                # speed = gps_tokens[7]

                # if heading == '':
                #     heading = None
                # else:
                #     heading = float(heading)
                # if speed == '':
                #     speed = None
                # else:
                #     speed = float(speed)

                if ref_lat is None:
                    ref_lat = latitude
                    ref_lon = longitude
                    ref_alt = altitude

                # Convert GPS to local ENU coordinates
                x, y = self.latlon_to_local(latitude, longitude, ref_lat, ref_lon)
                z = altitude - ref_alt
                gps_position = np.array([x, y, z])
                gps_positions.append(gps_position)

                # Parse pose data
                pose_tokens = pose_line.split(',')
                if len(pose_tokens) < 11:
                    continue
                qw = float(pose_tokens[2])
                qx = float(pose_tokens[3])
                qy = float(pose_tokens[4])
                qz = float(pose_tokens[5])
                tx = float(pose_tokens[6])
                ty = float(pose_tokens[7])
                tz = float(pose_tokens[8])
                image_name = pose_tokens[10]
                pose = [tx, ty, tz, qx, qy, qz, qw]
                poses.append(pose)
                images.append(image_name)

            gps_positions = np.array(gps_positions)
            poses = np.array(poses)
            if poses.shape[0] == 0 or gps_positions.shape[0] == 0:
                continue
            usable = poses.shape[0] - self.context_size - max(self.search_window, self.wp_length)
            self.count.append(max(usable, 0))
            self.gps_positions.append(gps_positions)
            self.poses.append(poses)
            self.image_names.append(images)

        valid_indices = [i for i, c in enumerate(self.count) if c > 0]
        self.gps_positions = [self.gps_positions[i] for i in valid_indices]
        self.poses = [self.poses[i] for i in valid_indices]
        self.image_names = [self.image_names[i] for i in valid_indices]
        self.image_folders = [self.image_folders[i] for i in valid_indices]
        self.count = [self.count[i] for i in valid_indices]

        self.lut = []
        self.sequence_ranges = []
        idx_counter = 0
        for seq_idx, count in enumerate(self.count):
            start_idx = idx_counter
            for pose_start in range(count):
                self.lut.append((seq_idx, pose_start))
                idx_counter += 1
            end_idx = idx_counter
            self.sequence_ranges.append((start_idx, end_idx))
        assert len(self.lut) > 0, "No usable samples found."

    def __len__(self):
        return len(self.lut)

    def __getitem__(self, index):
        sequence_idx, pose_start = self.lut[index]
        gps_positions = self.gps_positions[sequence_idx]
        poses = self.poses[sequence_idx]
        images = self.image_names[sequence_idx]
        image_folder = self.image_folders[sequence_idx]

        # Get input GPS positions
        input_gps_positions = gps_positions[pose_start: pose_start + self.context_size]
        future_gps_positions = gps_positions[pose_start + self.context_size: pose_start + self.context_size + self.search_window]
        if future_gps_positions.shape[0] == 0:
            raise IndexError(f"No future positions available for index {pose_start}.")

        # Select target GPS position
        target_idx, arrived = self.select_target_index(future_gps_positions)
        target_gps_position = future_gps_positions[target_idx]

        # Transform input GPS positions by subtracting target GPS position
        input_positions = self.input2target(input_gps_positions, target_gps_position)[:, [0, 1]]

        # Apply random rotation if in training mode
        if self.mode == 'train':
            rand_angle = np.random.uniform(-np.pi, np.pi)
            rot_matrix = np.array([[np.cos(rand_angle), -np.sin(rand_angle)],
                                   [np.sin(rand_angle), np.cos(rand_angle)]])
            input_positions = input_positions @ rot_matrix.T

        input_positions = torch.tensor(input_positions, dtype=torch.float32)
        arrived = torch.tensor(arrived, dtype=torch.float32)

        # Load frames
        input_image_names = images[pose_start: pose_start + self.context_size]
        frames = self.load_frames(image_folder, input_image_names)

        # Get waypoints from poses
        # For input frames (history positions)
        input_poses = poses[pose_start: pose_start + self.context_size]
        # For future frames (gt waypoints)
        waypoint_start = pose_start + self.context_size
        waypoint_end = waypoint_start + self.wp_length
        gt_waypoint_poses = poses[waypoint_start: waypoint_end]

        # Transform waypoints to the coordinate frame of the current pose
        current_pose = input_poses[-1]
        history_positions = self.transform_poses(input_poses, current_pose)
        gt_waypoints = self.transform_poses(gt_waypoint_poses, current_pose)

        # Select target pose for visualization
        target_pose = poses[pose_start + self.context_size + target_idx]
        target_transformed = self.transform_pose(target_pose, current_pose)

        # Convert to tensors
        waypoints_transformed = torch.tensor(gt_waypoints[:, [0, 1]], dtype=torch.float32)

        sample = {
            'video_frames': frames,
            'input_positions': input_positions,
            'waypoints': waypoints_transformed,
            'arrived': arrived
        }
        print("input", input_positions)
        print("history", history_positions)
        print("wp", waypoints_transformed)
        print("target", target_transformed)

        if self.mode in ['val', 'test']:
            # For visualization
            history_positions = torch.tensor(history_positions[:, [0, 1]], dtype=torch.float32)
            target_transformed_position = torch.tensor(target_transformed[[0, 1]], dtype=torch.float32)

            sample['original_input_positions'] = history_positions
            sample['noisy_input_positions'] = history_positions
            sample['gt_waypoints'] = waypoints_transformed
            sample['target_transformed'] = target_transformed_position

        return sample

    def input2target(self, input_positions, target_position):
        transformed_input_positions = input_positions - target_position
        return transformed_input_positions

    def select_target_index(self, future_positions):
        arrived = np.random.rand() < self.arrived_prob
        max_idx = future_positions.shape[0] - 1
        if arrived:
            target_idx = random.randint(self.wp_length, min(self.wp_length + self.arrived_threshold, max_idx))
        else:
            target_idx = random.randint(self.wp_length + self.arrived_threshold, max_idx)
        return target_idx, arrived

    def transform_poses(self, poses, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrices = self.poses_to_matrices(poses)
        transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        positions = transformed_matrices[:, :3, 3]
        return positions

    def transform_pose(self, pose, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrix = self.pose_to_matrix(pose)
        transformed_matrix = np.matmul(current_pose_inv, pose_matrix)
        position = transformed_matrix[:3, 3]
        return position

    def pose_to_matrix(self, pose):
        tx, ty, tz, qx, qy, qz, qw = pose
        rotation = R.from_quat([qx, qy, qz, qw])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = [tx, ty, tz]
        return matrix

    def poses_to_matrices(self, poses):
        tx = poses[:, 0]
        ty = poses[:, 1]
        tz = poses[:, 2]
        qx = poses[:, 3]
        qy = poses[:, 4]
        qz = poses[:, 5]
        qw = poses[:, 6]
        rotations = R.from_quat(np.stack([qx, qy, qz, qw], axis=1))
        matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
        matrices[:, :3, :3] = rotations.as_matrix()
        matrices[:, :3, 3] = np.stack([tx, ty, tz], axis=1)
        return matrices

    def load_frames(self, image_folder, image_names):
        frames = []
        for image_name in image_names:
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} does not exist.")
            image = Image.open(image_path).convert('RGB')
            image = TF.to_tensor(image)
            frames.append(image)
        frames = torch.stack(frames)
        return frames

    def latlon_to_local(self, lat, lon, lat0, lon0):
        R_earth = 6378137  # Earth's radius in meters
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        dlat = lat_rad - lat0_rad
        dlon = lon_rad - lon0_rad
        x = dlon * np.cos((lat_rad + lat0_rad) / 2) * R_earth
        y = dlat * R_earth
        return x, y