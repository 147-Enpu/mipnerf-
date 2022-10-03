import numpy as np
import copy
import open3d as o3d
import os
import torch
import glob
import imageio
from nerf_sample_ray_split import RaySamplerSingleImage
from collections import OrderedDict

def get_center_and_diag(cam_centers):
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal

def transform_pose(C2W, translate, scale):
    cam_center = C2W[:3, 3]
    # print("cam_center:",cam_center)
    # print("translate",translate)
    # print("scale:",scale)
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    return C2W

def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        print("Not GOOOOOOOOOOD")
    else:
        print("Looks good")
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos


    return d1 + d2

def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

def parse_txt(filename):
    assert os.path.isfile(filename)
    nums = open(filename).read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

pose_files = sorted(os.listdir(os.path.join('pose_out2')))
cam_centers = []
target_radius = 1.
for pose_name in pose_files:
    C2W = np.loadtxt(os.path.join('pose_out2',pose_name))
    C2W = np.array(C2W).reshape((4, 4))
    #print("C2W1:",C2W[:3, 3])
    cam_centers.append(C2W[:3, 3:4])

center, diagonal = get_center_and_diag(cam_centers)
radius = diagonal * 1

translate = -center
scale = target_radius / radius

for pose_name in pose_files:
    C2W = np.loadtxt(os.path.join('pose_out2',pose_name))
    C2W = np.array(C2W).reshape((4, 4))
    C2W = transform_pose(C2W, translate, scale)
    lines = [
        "{} {} {} {} ".format(C2W[0, 0], C2W[0, 1], C2W[0, 2], C2W[0, 3]),
        "{} {} {} {} ".format(C2W[1, 0], C2W[1, 1], C2W[1, 2], C2W[1, 3]),
        "{} {} {} {} ".format(C2W[2, 0], C2W[2, 1], C2W[2, 2], C2W[2, 3]),
        "0.0 0.0 0.0 1.0 ",
    ]
    with open(os.path.join(os.getcwd(), "pose2", "%s.txt" % pose_name.split('.', 1)[0]), "w+") as f:
        f.writelines(lines)
    print("C2W2:",C2W[1,3])


####Check
intrinsics_files = find_files(os.path.join(os.getcwd(), "intrinsics_out2"), exts=['*.txt'])
pose_files = find_files('pose2', exts=['*.txt'])
intrinsics_files = intrinsics_files[::1]
pose_files = pose_files[::1]
cam_cnt = len(pose_files)
img_files = find_files('/home/enpu/nerfplusplus/out7/posed_images/images', exts=['*.png', '*.jpg'])
img_files = img_files[::1]

img_files = [None, ] * cam_cnt

train_imgfile = find_files('/home/enpu/nerfplusplus/out7/posed_images/images', exts=['*.png', '*.jpg'])[0]
train_im = imageio.imread(train_imgfile)
H, W = train_im.shape[:2]

# create ray samplers
ray_samplers = []
for i in range(cam_cnt):
    intrinsics = parse_txt(intrinsics_files[i])
    pose = parse_txt(pose_files[i])

    ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                              img_path=img_files[i],
                                              mask_path=None,
                                              min_depth_path=None,
                                              max_depth=None))

# print(ray_samplers)
# ray_batch = ray_samplers[i].random_sample(32 * 32 * 2, center_crop=False)
# fg_far_depth = intersect_sphere(ray_batch['ray_o'], ray_batch['ray_d'])

for idx in range(len(ray_samplers)):
    world_size = -1
    ray_batch = ray_samplers[idx].get_all()
    # rank_split_sizes = [ray_batch['ray_d'].shape[0] // world_size, ] * world_size
    # rank_split_sizes[-1] = ray_batch['ray_d'].shape[0] - sum(rank_split_sizes[:-1])
    # for key in ray_batch:
    #     if torch.is_tensor(ray_batch[key]):
    #         ray_batch[key] = torch.split(ray_batch[key], rank_split_sizes)

    ray_batch_split = OrderedDict()
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch_split[key] = torch.split(ray_batch[key], 1024 * 8)

    for s in range(len(ray_batch_split['ray_d'])):
        ray_o = ray_batch_split['ray_o'][s]
        ray_d = ray_batch_split['ray_d'][s]
        fg_far_depth = intersect_sphere(ray_o, ray_d)