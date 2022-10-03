import torch
import torch.nn as nn
from collections import OrderedDict
from ray_utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map
from pose_utils import to8b
from nerf_network import Embedder, MLPNet
from utils import TINY_NUMBER, HUGE_NUMBER

######################################################################################
# wrapper to simplify the use of nerfnet
######################################################################################
def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real


class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None]**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return x_ret


class MipNeRF(nn.Module):
    def __init__(self,
                 args,
                 use_viewdirs=True,
                 randomized=False,
                 ray_shape="cone",
                 white_bkgd=False,
                 num_levels=2,
                 num_samples=128,
                 hidden=256,
                 density_noise=1,
                 density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 min_deg=0,
                 max_deg=16,
                 viewdirs_min_deg=0,
                 viewdirs_max_deg=4,
                 return_raw=False,
                 netdepth=8,
                 netwidth=256,
                 ):
        super(MipNeRF, self).__init__()
        self.use_viewdirs = use_viewdirs
        self.init_randomized = randomized
        self.randomized = randomized
        self.ray_shape = ray_shape
        self.white_bkgd = white_bkgd
        self.num_levels = num_levels
        self.num_samples = num_samples
        self.density_input = (max_deg - min_deg) * 3 * 2
        self.rgb_input = 3 + ((viewdirs_max_deg - viewdirs_min_deg) * 3 * 2)
        self.density_noise = density_noise
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.density_bias = density_bias
        self.hidden = hidden
        #self.device = device
        self.return_raw = return_raw
        self.netdepth = netdepth
        self.netwidth = netwidth
        self.density_activation = nn.Softplus()

        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.bg_net = MLPNet(D=self.netdepth, W=self.netwidth,
                             input_ch=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)

        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.density_net0 = nn.Sequential(
            nn.Linear(self.density_input, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.density_net1 = nn.Sequential(
            nn.Linear(self.density_input + hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.final_density = nn.Sequential(
            nn.Linear(hidden, 1),
        )

        input_shape = hidden
        if self.use_viewdirs:
            input_shape = num_samples

            self.rgb_net0 = nn.Sequential(
                nn.Linear(hidden, hidden)
            )
            self.viewdirs_encoding = PositionalEncoding(viewdirs_min_deg, viewdirs_max_deg)
            self.rgb_net1 = nn.Sequential(
                nn.Linear(hidden + self.rgb_input, num_samples),
                nn.ReLU(True),
            )
        self.final_rgb = nn.Sequential(
            nn.Linear(input_shape, 3),
            #nn.Sigmoid()
        )
        _xavier_init(self)
        #self.to(device)

    def forward(self, ray_o, ray_d, fg_t_vals, bg_z_vals, fg_mean, fg_var):
        comp_rgbs = []
        distances = []
        accs = []
        #for l in range(self.num_levels):
            # sample
            # if l == 0:  # coarse grain sample
            #     t_vals, (mean, var) = sample_along_rays(rays.origins, rays.directions, rays.radii, self.num_samples,
            #                                             rays.near, rays.far, randomized=self.randomized, lindisp=False,
            #                                             ray_shape=self.ray_shape)
            # else:  # fine grain sample/s
            #     t_vals, (mean, var) = resample_along_rays(rays.origins, rays.directions, rays.radii,
            #                                               t_vals.to(rays.origins.device),
            #                                               weights.to(rays.origins.device), randomized=self.randomized,
            #                                               stop_grad=True, resample_padding=self.resample_padding,
            #                                               ray_shape=self.ray_shape)
            # do integrated positional encoding of samples
        fg_samples_enc = self.positional_encoding(fg_mean, fg_var)[0]
        fg_samples_enc = fg_samples_enc.reshape([-1, fg_samples_enc.shape[-1]])

        # predict density
        fg_new_encodings = self.density_net0(fg_samples_enc)
        fg_new_encodings = torch.cat((fg_new_encodings, fg_samples_enc), -1)
        fg_new_encodings = self.density_net1(fg_new_encodings)
        fg_raw_density = self.final_density(fg_new_encodings).reshape((-1, self.num_samples, 1))

        # predict rgb
        if self.use_viewdirs:
            #  do positional encoding of viewdirs
            ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
            viewdirs = ray_d / ray_d_norm  # [..., 3]

            fg_viewdirs = self.viewdirs_encoding(viewdirs)
            fg_viewdirs = torch.cat((fg_viewdirs, viewdirs), -1)
            fg_viewdirs = torch.tile(fg_viewdirs[:, None, :], (1, self.num_samples, 1))
            fg_viewdirs = fg_viewdirs.reshape((-1, fg_viewdirs.shape[-1]))
            fg_new_encodings = self.rgb_net0(fg_new_encodings)
            # print("fg_new_encodings:", fg_new_encodings.shape)
            # print("fg_viewdirs:", fg_viewdirs.shape)
            fg_new_encodings = torch.cat((fg_new_encodings, fg_viewdirs), -1)
            fg_new_encodings = self.rgb_net1(fg_new_encodings)
        fg_raw_rgb = self.final_rgb(fg_new_encodings).reshape((-1, self.num_samples, 3))

        # Add noise to regularize the density predictions if needed.
        if self.randomized and self.density_noise:
            fg_raw_density += self.density_noise * torch.rand(fg_raw_density.shape, dtype=fg_raw_density.dtype, device=fg_raw_density.device)

        # volumetric rendering
        fg_rgb = fg_raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        fg_density = self.density_activation(fg_raw_density + self.density_bias)
        fg_comp_rgb, fg_distance, fg_acc, fg_weights, fg_alpha, bg_lambda = volumetric_rendering(fg_rgb, fg_density, fg_t_vals, ray_d, self.white_bkgd)
        # comp_rgbs.append(comp_rgb)
        # distances.append(distance)
        # accs.append(acc)

        # render background
        N_samples = bg_z_vals.shape[-1]
        dots_sh = list(ray_d.shape[:-1])
        bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_pts, _ = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        input = torch.cat((self.bg_embedder_position(bg_pts),
                           self.bg_embedder_viewdir(bg_viewdirs)), dim=-1)
        # near_depth: physical far; far_depth: physical near
        input = torch.flip(input, dims=[-2, ])
        bg_z_vals = torch.flip(bg_z_vals, dims=[-1, ])  # 1--->0
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        bg_raw = self.bg_net(input)
        bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # Eq. (3): T
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        bg_weights = bg_alpha * T  # [..., N_samples]
        bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
        bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

        # composite foreground and background
        bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map
        bg_depth_map = bg_lambda * bg_depth_map
        rgb_map = fg_comp_rgb + bg_rgb_map

        ret = OrderedDict([('rgb', rgb_map),  # loss
                           ('fg_weights', fg_weights),  # importance sampling
                           ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb', fg_comp_rgb),  # below are for logging
                           ('bg_rgb', bg_rgb_map),
                           ('bg_depth', bg_depth_map),
                           ('bg_lambda', bg_lambda)])
        #
        # ret = OrderedDict([('fg_weights', fg_weights),  # importance sampling
        #                    ('fg_rgb', fg_comp_rgb)])
        return ret

        # if self.return_raw:
        #     raws = torch.cat((torch.clone(rgb).detach(), torch.clone(density).detach()), -1).cpu()
        #     # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
        #     return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
        # else:
        #     # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
        #     return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)

    def render_image(self, rays, height, width, chunks=8192):
        """
        Return image, disparity map, accumulated opacity (shaped to height x width) created using rays as input.
        Rays should be all of the rays that correspond to this one single image.
        Batches the rays into chunks to not overload memory of device
        """
        length = rays[0].shape[0]
        rgbs = []
        dists = []
        accs = []
        with torch.no_grad():
            for i in range(0, length, chunks):
                # put chunk of rays on device
                chunk_rays = namedtuple_map(lambda r: r[i:i+chunks].to(self.device), rays)
                rgb, distance, acc = self(chunk_rays)
                rgbs.append(rgb[-1].cpu())
                dists.append(distance[-1].cpu())
                accs.append(acc[-1].cpu())

        rgbs = to8b(torch.cat(rgbs, dim=0).reshape(height, width, 3).numpy())
        dists = torch.cat(dists, dim=0).reshape(height, width).numpy()
        accs = torch.cat(accs, dim=0).reshape(height, width).numpy()
        return rgbs, dists, accs

    def train(self, mode=True):
        self.randomized = self.init_randomized
        super().train(mode)
        return self

    def eval(self):
        self.randomized = False
        return super().eval()


def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
