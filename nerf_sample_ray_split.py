import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio

########################################################################################################################
# ray batch sampling
########################################################################################################################
def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    return rays_o, rays_d, depth


class RaySamplerSingleImage(object):
    def __init__(self, H, W, intrinsics, c2w,
                       img_path=None,
                       img_nobg=None,
                       resolution_level=1,
                       mask_path=None,
                       min_depth_path=None,
                       max_depth=None):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics
        self.c2w_mat = c2w

        self.img_path = img_path
        self.img_nobg = img_nobg
        self.mask_path = mask_path
        self.min_depth_path = min_depth_path
        self.max_depth = max_depth

        self.resolution_level = -1
        self.set_resolution_level(resolution_level)

        #self.cam_to_world = np.stack(cams, axis=0)

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level
            # only load image at this time
            if self.img_path is not None:
                self.img = imageio.imread(self.img_path).astype(np.float32) / 255.
                self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.img = self.img.reshape((-1, 3))
            else:
                self.img = None

            if self.img_nobg is not None:
                self.img_nb = imageio.imread(self.img_nobg).astype(np.float32) / 255.
                self.img_nb = cv2.resize(self.img_nb, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.img_nb = self.img.reshape((-1, 3))
            else:
                self.img_nb = None

            if self.mask_path is not None:
                self.mask = imageio.imread(self.mask_path).astype(np.float32) / 255.
                self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                self.mask = self.mask.reshape((-1))
            else:
                self.mask = None

            if self.min_depth_path is not None:
                self.min_depth = imageio.imread(self.min_depth_path).astype(np.float32) / 255. * self.max_depth + 1e-4
                self.min_depth = cv2.resize(self.min_depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self.min_depth = self.min_depth.reshape((-1))
            else:
                self.min_depth = None

            self.rays_o, self.rays_d, self.depth = get_rays_single_image(self.H, self.W,
                                                                         self.intrinsics, self.c2w_mat)

    def get_img(self):
        if self.img is not None:
            return self.img.reshape((self.H, self.W, 3))
        else:
            return None

    def get_all(self):
        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])

        dx = np.sqrt(np.sum((self.rays_d[:-1, :] - self.rays_d[1:, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[-2:-1]])
        # print("dx:", dx.shape)

        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / np.sqrt(12)

        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('radii', radii),
            ('depth', self.depth),
            ('rgb', self.img),
            ('mask', self.mask),
            ('min_depth', min_depth)
        ])
        # return torch tensors
        for k in ret:
            if ret[k] is not None:
                ret[k] = torch.from_numpy(ret[k])
        return ret

    def random_sample(self, N_rand, center_crop=False):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if center_crop:
            half_H = self.H // 2
            half_W = self.W // 2
            quad_H = half_H // 2
            quad_W = half_W // 2

            # pixel coordinates
            u, v = np.meshgrid(np.arange(half_W-quad_W, half_W+quad_W),
                               np.arange(half_H-quad_H, half_H+quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

            # Convert back to original image
            select_inds = v[select_inds] * self.W + u[select_inds]
        else:
            # Random from one image
            select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)

        rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
        depth = self.depth[select_inds]         # [N_rand, ]

        if self.img is not None:
            rgb = self.img[select_inds, :]          # [N_rand, 3]
        else:
            rgb = None

        if self.img_nb is not None:
            rgb_nb = self.img_nb[select_inds, :]          # [N_rand, 3]
        else:
            rgb_nb = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        if self.min_depth is not None:
            min_depth = self.min_depth[select_inds]
        else:
            min_depth = 1e-4 * np.ones_like(rays_d[..., 0])

        x, y = np.meshgrid(
            np.arange(self.W, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.H, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')

        #focal = self.intrinsics[0,0]
        #print("focal:",focal)
        # camera_directions = np.stack(
        #     [(x - self.W * 0.5 + 0.5) / focal,
        #      -(y - self.H * 0.5 + 0.5) / focal,
        #      -np.ones_like(x)],
        #     axis=-1)

        #directions = ((camera_directions[None, ..., None, :] * self.cam_to_world[:, None, None, :3, :3]).sum(axis=-1))  # Translate camera frame's origin to the world frame
        # origins = np.broadcast_to(self.cam_to_world[:, None, None, :3, -1], directions.shape)
        # viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        # Distance from each unit-norm direction vector to its x-axis neighbor
        # print("rays_o:",rays_o.shape)
        # print("rays_d:", rays_d.shape)
        # print("self.rays_o:",self.rays_o.shape)
        # print("self.rays_d:", self.rays_d.shape)
        # rays_o: (1024, 3)
        # rays_d: (1024, 3)
        # self.rays_o: (793078, 3)
        # self.rays_d: (793078, 3)

        dx = np.sqrt(np.sum((self.rays_d[:-1, :] - self.rays_d[1:, :]) ** 2, -1))
        # dx = np.sqrt(np.sum((rays_o[:-1, :] - rays_o[1:, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[-2:-1]])
        #print("dx:", dx.shape)

        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / np.sqrt(12)
        radii = radii[select_inds, :]
        # print("radii",radii)

        # print("radii:",radii.shape)
        # print("rays_d:",rays_d.shape)

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('radii', radii),
            ('depth', depth),
            ('rgb', rgb),
            ('rgb_nb', rgb_nb),
            ('mask', mask),
            ('min_depth', min_depth),
            ('img_name', self.img_path)
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret
