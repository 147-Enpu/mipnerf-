import numpy as np
import torch
import os

def circle(radius=3.5, h=30.0, axis='y', t0=0, r=1):
    if axis == 'z':
        return lambda t: [radius * np.cos(r * t + t0), radius * np.sin(r * t + t0), h]
    elif axis == 'y':
        #return lambda t: [radius * np.sin(r * t + t0), 0.87*(radius * np.cos(r * t + t0)), 0.5*(-radius * np.cos(r * t + t0)-30)]
        #return lambda t: [radius * np.cos(r * t + t0), h, radius * np.sin(r * t + t0)]
        return lambda t: [radius * np.sin(r * t + t0), 0.87*radius * np.cos(r * t + t0), -0.5*radius * np.cos(r * t + t0)-10]
        #return lambda t: [h, radius * np.cos(r * t + t0), radius * np.sin(r * t + t0)]
    else:
        return lambda t: [h, radius * np.cos(r * t + t0), radius * np.sin(r * t + t0)]

def gen_path(pos_gen, at=(0, 0, 0), up=(0, 1, 0), frames=180):
    c2ws = []
    for t in range(frames):
        c2w = torch.eye(4)
        cam_pos = torch.tensor(pos_gen(t * (360.0 / frames) / 180 * np.pi))
        # print("cam_pos:",cam_pos)
        at=(1, 0, 0)
        #up=(0.5, 1, -0.5)
        cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
        # print("up:",up)
        # print("cam_rot:",cam_rot)
        c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
        c2ws.append(c2w)
    return torch.stack(c2ws)

def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = torch.zeros_like(camera_position)
    else:
        at = torch.tensor(at).type_as(camera_position)
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    print("at:",at)
    print("up:",up)
    print("camera_position:",camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(up, z_axis))[0]
    y_axis = normalize(cross(z_axis, x_axis))[0]

    R = cat([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R

def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)

def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2 == 0] = 1
        return x / l2,

def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)


pose_files = sorted(os.listdir(os.path.join('tanks', 'pose')))
scene_bbox = torch.from_numpy(np.loadtxt(os.path.join('gt', 'bbox.txt'))).float()[:6].view(2,3)*1.2
poses = []
for pose_fname in pose_files:
    c2w = np.loadtxt(os.path.join('tanks', 'pose', pose_fname))# @ cam_trans
    c2w = c2w.reshape(4,4)
    c2w = torch.FloatTensor(c2w)
    poses.append(c2w)
#print(poses)

poses = torch.stack(poses)
center = torch.mean(scene_bbox, dim=0)
radius = torch.norm(scene_bbox[1]-center)*2
#print("radius:",radius)
up = torch.mean(poses[:, :3, 1], dim=0).tolist()
print("up:",up[1])
pos_gen = circle(radius=radius, h=-0.2*up[1], axis='y')
#print("pos_gen:",pos_gen)
up = (0, 1, 1)
render_path = gen_path(pos_gen, up=up,frames=200)
render_path[:, :3, 3] += center


if not os.path.exists(os.path.join(os.getcwd(), "pose_out2")):
    os.mkdir(os.path.join(os.getcwd(), "pose_out2"))
if not os.path.exists(os.path.join(os.getcwd(), "intrinsics_out2")):
    os.mkdir(os.path.join(os.getcwd(), "intrinsics_out2"))

par = 0
for i in range(200):
    #print(render_path[i])
    pose = render_path[i]
    lines = [
        "{} {} {} {} ".format(pose[0,0], pose[0,1], pose[0,2], pose[0,3]),
        "{} {} {} {} ".format(pose[1,0], pose[1,1], pose[1,2], pose[1,3]),
        "{} {} {} {} ".format(pose[2,0], pose[2,1], pose[2,2], pose[2,3]),
        "0.0 0.0 0.0 1.0",
    ]
    with open(os.path.join(os.getcwd(), "pose_out2", "%06d.txt" % par), "w+") as f:
        f.writelines(lines)
    par += 1

#print(render_path.shape)

intrinsic = np.loadtxt(os.path.join('tanks', 'intrinsics', "000001.txt"))
#print(intrinsic)
par_i = 0
for i in range(200):
    lines = [
        "{} {} {} {} ".format(intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]),
        "{} {} {} {} ".format(intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7]),
        "{} {} {} {} ".format(intrinsic[8], intrinsic[9], intrinsic[10], intrinsic[11]),
        "0.0 0.0 0.0 1.0",
    ]
    with open(os.path.join(os.getcwd(), "intrinsics_out2", "%06d.txt" % par_i), "w+") as f:
        f.writelines(lines)
    par_i += 1