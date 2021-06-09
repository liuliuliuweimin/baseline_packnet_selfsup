import torch
import os
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from packnet_sfm.utils.types import is_tensor, is_numpy, is_list
from packnet_sfm.utils.misc import make_list
from dgp.utils.geometry import Pose

def stack_sample(sample):
    """Stack a sample from multiple sensors"""
    # If there is only one sensor don't do anything
    if len(sample) == 1:
        return sample[0]

    # Otherwise, stack sample
    stacked_sample = {}
    for key in sample[0]:
        # Global keys (do not stack)
        if key in ['idx', 'dataset_idx', 'sensor_name', 'filename']:
            stacked_sample[key] = sample[0][key]
        else:
            # Stack torch tensors
            if is_tensor(sample[0][key]):
                stacked_sample[key] = torch.stack([s[key] for s in sample], 0)
            # Stack numpy arrays
            elif is_numpy(sample[0][key]):
                stacked_sample[key] = np.stack([s[key] for s in sample], 0)
            # Stack list
            elif is_list(sample[0][key]):
                stacked_sample[key] = []
                # Stack list of torch tensors
                if is_tensor(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            torch.stack([s[key][i] for s in sample], 0))
                # Stack list of numpy arrays
                if is_numpy(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            np.stack([s[key][i] for s in sample], 0))

    # Return stacked sample
    return stacked_sample


##

class NuScenesDataset(Dataset):
    def __init__(self, path, split,
                 cameras=None,
                 depth_type=None,
                 with_pose=False,
                 back_context=0,
                 forward_context=0,
                 data_transform=None):
        super().__init__()
        self.meta_image_path = '{}/spatial_temp_{}_v2.json'.format(path, split)
        self.meta_depth_path = '{}/two_cam_meta/spatial_temp_merged_path_{}.json'.format(path, split)
        self.meta_mat_path = '{}/cam_pose_intrinsic_v3.json'.format(path)
        self.meta_data_dir = path  # save for use in the future
        self.data_dir = path.replace('-SF/meta', '')  # data_dir, .../nuScenes
        self.split = split
        self.dataset_idx = 0
        self.bwd = back_context
        self.fwd = forward_context
        self.has_context = back_context + forward_context > 0
        self.num_cameras = len(cameras)
        self.data_transform = data_transform

        self.depth_type = depth_type
        self.with_depth = depth_type is not None
        self.with_pose = with_pose

        self.load_annotations()

    def get_current(self, key, cam_id):
        if key == 'datum_name':
            return os.path.split(os.path.dirname(self.sample_nuscenes['now']['filename'][cam_id]))[1].lower()
        elif key == 'rgb':
            return Image.open(os.path.join(self.data_dir, self.sample_nuscenes['now']['filename'][cam_id])).convert(
                'RGB')
        elif key == 'intrinsics':
            token = self.sample_nuscenes['now']['cam_token'][cam_id]
            return np.array(self.meta_mat_dataset[token]['intrinsic'])
        elif key == 'depth':
            depth_path = os.path.join(self.data_dir, self.meta_depth_dataset[self.sample_nuscenes['now']
            ['filename'][cam_id]]['depth_path'].replace('data/nuscenes/',''))+'.npy'
            depth_points = np.load(depth_path)
            return self._generate_depth_map(depth_points)
        elif key == 'extrinsics':
            token = self.sample_nuscenes['now']['cam_token'][cam_id]
            return np. array(self.meta_mat_dataset[token]['pose'])
        elif key == 'pose':
            token = self.sample_nuscenes['now']['cam_token'][cam_id]
            return np.array(self.meta_mat_dataset[token]['pose'])
        else:
            raise NotImplementedError

    def get_context(self,key, cam_id):
        if key == 'rgb':
            return [Image.open(os.path.join(self.data_dir, self.sample_nuscenes['prev']['filename'][cam_id])).convert(
                'RGB'),Image.open(os.path.join(self.data_dir, self.sample_nuscenes['next']['filename'][cam_id])).convert(
                'RGB')]
        elif key == 'extrinsics':
            token_pre = self.sample_nuscenes['prev']['cam_token'][cam_id]
            token_next = self.sample_nuscenes['next']['cam_token'][cam_id]
            return [np.array(self.meta_mat_dataset[token_pre]['pose']),
                    np.array(self.meta_mat_dataset[token_next]['pose'])]
        elif key == 'pose':
            token_pre = self.sample_nuscenes['prev']['cam_token'][cam_id]
            token_next = self.sample_nuscenes['next']['cam_token'][cam_id]
            return [np.array(self.meta_mat_dataset[token_pre]['pose']),
                    np.array(self.meta_mat_dataset[token_next]['pose'])]
        else:
            raise NotImplementedError

    def get_filename(self,idx,cam_id):
        return self.meta_image_dataset[idx]['now']['filename'][cam_id]


    def _sort_points(self, points):
        '''
        sort the points accroding to their depth in descending order
        '''
        depth = points[:, 2]
        idx = np.argsort(depth) # ascending order
        idx = idx[::-1]

        new_points = points[idx]

        return new_points

    def _generate_depth_map(self, points, shape=(900,1600)):
        '''
        for float cord, use its int version
        '''
        depth_map = np.zeros(shape)
        points = self._sort_points(points)

        x_cords = points[:, 0]
        y_cords = points[:, 1]
        depth = points[:, 2]
        depth = np.clip(depth, a_min=1e-5, a_max=99999)

        # print('debug', depth[:10].mean(), depth[10:100].mean(), depth[-100:].mean())
        # print('debug', x_cords.max(), y_cords.max(), depth_map.shape)
        x_cords = x_cords.astype(np.int)
        y_cords = y_cords.astype(np.int)

        # first y, then x
        # print(depth_map.shape, )
        depth_map[y_cords, x_cords] = depth

        return depth_map.astype(np.float32)



    def load_annotations(self):
        with open(self.meta_image_path, 'r') as f:
            self.meta_image_dataset = json.load(f)  # List [dict{'prev','now','next'}*N]
            # 'now':dict{'filename','cam_token'}
            # 'filename': List [filename*6]
        if self.with_depth:
            with open(self.meta_depth_path, 'r') as f:
                self.meta_depth_dataset = json.load(f)  # dict {'*filename'*N} /samples/CAM_FRONT/_.jpg
        if self.with_pose:
            with open(self.meta_mat_path, 'r') as f:
                self.meta_mat_dataset = json.load(f)  # dict {*'cam_token'}
        # token = self.meta_image_dataset[0]['now']['cam_token'][0]
        # print(self.meta_mat_dataset[token])
        # raise NotImplementedError

    def __len__(self):
        return len(self.meta_image_dataset)

    def __getitem__(self, idx):
        """Get a dataset sample"""
        # Get DGP sample (if single sensor, make it a list)
        # Loop over all cameras
        self.sample_nuscenes = self.meta_image_dataset[idx]
        sample = []
        for i in range(self.num_cameras):
            data = {
                'idx': idx,
                'dataset_idx': self.dataset_idx,
                'sensor_name': self.get_current('datum_name', i),
                #
                'filename': self.get_filename(idx, i),
                'splitname': '%s_%010d' % (self.split, idx),
                #
                'rgb': self.get_current('rgb', i),
                'intrinsics': self.get_current('intrinsics', i),
            }
            # If depth is returned
            if self.with_depth:
                data.update({
                    'depth': self.get_current('depth',i)
                })

            # If pose is returned
            if self.with_pose:
                data.update({
                    'extrinsics': self.get_current('extrinsics', i),
                    'pose': self.get_current('pose', i),
                })

            # If context is returned
            if self.has_context:
                data.update({
                    'rgb_context': self.get_context('rgb', i),
                })
                # If context pose is returned
                if self.with_pose:
                    # Get original values to calculate relative motion
                    # print(data['extrinsics'])
                    orig_extrinsics = Pose.from_matrix(data['extrinsics'])
                    orig_pose = Pose.from_matrix(data['pose'])
                    data.update({
                        'extrinsics_context':
                            [(orig_extrinsics.inverse() * Pose.from_matrix(extrinsics)).matrix
                             for extrinsics in self.get_context('extrinsics', i)],
                        'pose_context':
                            [(orig_pose.inverse() * Pose.from_matrix(pose)).matrix
                             for pose in self.get_context('pose', i)],
                    })

            sample.append(data)

            # Apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]

            # Return sample (stacked if necessary)
        return stack_sample(sample)
