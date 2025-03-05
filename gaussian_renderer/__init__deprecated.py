#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import warnings
import random
import torch
import functorch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
from utils.graphics_utils import fov2focal
import torch.nn.functional as F
from kornia.geometry import depth_to_normals
from torchvision.utils import save_image


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


'''
def torch_render(pc, camera, means2D, cov2d, color, opacity, mean_view, bg, normal_d, normal):
    if (bg.cpu() == torch.tensor([1,1,1])).all():
        white_bkgd = True
    elif (bg.cpu() == torch.tensor([0,0,0])).all():
        white_bkgd = False
    else:
        assert False

    depths = mean_view[:,2]

    radii = get_radius(cov2d)
    rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
    
    render_color = torch.ones(*pc.pix_coord.shape[:2], 3).to('cuda')
    render_depth = torch.zeros(*pc.pix_coord.shape[:2], 1).to('cuda')
    render_alpha = torch.zeros(*pc.pix_coord.shape[:2], 1).to('cuda')

    TILE_SIZE = 50
    sorted_rets = []
    dd_loss = 0
    nc_loss = 0
    tile_num = (camera.image_height/TILE_SIZE)*(camera.image_width/TILE_SIZE)
    for h in range(0, camera.image_height, TILE_SIZE):
        for w in range(0, camera.image_width, TILE_SIZE):
            # check if the rectangle penetrate the tile
            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
            
            if not in_mask.sum() > 0:
                continue

            P = in_mask.sum()
            tile_coord = pc.pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
            sorted_depths, index = torch.sort(depths[in_mask])
            sorted_normal = normal[in_mask][index]

            with torch.no_grad():
                #sorted_mean_view = mean_view[in_mask][index]
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index] # P 2 2
                sorted_conic = sorted_cov2d.inverse() # inverse of variance
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]
                dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
                
                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
                
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)

                ###
                acc_alpha = (alpha * T).sum(dim=1)
                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if white_bkgd else 0)
                tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
                render_color[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
                render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)
                ###

                weights = (alpha*T).squeeze(-1).detach()
                inds = torch.randperm(weights.shape[0])
                #inds = inds[:10]
                ray_inds = torch.randperm(weights.shape[1])
                ray_inds = ray_inds[:20]

                weights_sel = weights[inds][:, ray_inds]
                weights_col = weights_sel.unsqueeze(-1)
                weights_row = weights_sel.unsqueeze(-2)
                weights_comb = weights_col * weights_row


            depths_sel = sorted_depths[ray_inds]
            d_col = depths_sel.unsqueeze(1)
            d_row = depths_sel.unsqueeze(0)
            d_comb = torch.abs(d_col - d_row)[None]

            dd_loss += (weights_comb*d_comb).mean() / int(tile_num)

            t_normal_d = normal_d[:,h:h+TILE_SIZE, w:w+TILE_SIZE].permute(1,2,0).reshape(-1,3)[inds]
            norm = (sorted_normal[ray_inds].unsqueeze(0) * t_normal_d.unsqueeze(1)).sum(-1)
            nc_loss += (weights_sel*(1-norm)).mean() / int(tile_num)

            #sorted_rets.append((alpha*T, sorted_depths, sorted_normal, normal_d[:,h:h+TILE_SIZE, w:w+TILE_SIZE]))
    sorted_rets = [dd_loss, nc_loss]

    #save_image(render_color.permute(2,0,1), 'test1/r.png') 
    #save_image(render_depth.permute(2,0,1).repeat(3,1,1), 'test1/d.png') 
    #save_image(render_alpha.permute(2,0,1).repeat(3,1,1), 'test1/o.png') 


    return sorted_rets
'''


'''
def torch_render2(pc, camera, means2D, cov2d, color, opacity, mean_view, bg, normal_d, normal):
    if (bg.cpu() == torch.tensor([1,1,1])).all():
        white_bkgd = True
    elif (bg.cpu() == torch.tensor([0,0,0])).all():
        white_bkgd = False
    else:
        assert False

    depths = mean_view[:,2]
        #sorted_depths, index = torch.sort(depths[in_mask])

    depths, index = torch.sort(depths)
    means2D = means2D[index]
    cov2d = cov2d[index]
    color = color[index]
    opacity = opacity[index]
    #mean_view = mean_view[index]
    normal = normal[index]

    radii = get_radius(cov2d)
    rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
    
    #render_color = torch.ones(*pc.pix_coord.shape[:2], 3).to('cuda')
    #render_depth = torch.zeros(*pc.pix_coord.shape[:2], 1).to('cuda')
    #render_alpha = torch.zeros(*pc.pix_coord.shape[:2], 1).to('cuda')

    TILE_SIZE = 25
    sorted_rets = []
    tile_num = (camera.image_height/TILE_SIZE)*(camera.image_width/TILE_SIZE)
    
    h = torch.arange(0, camera.image_height, TILE_SIZE).to("cuda")
    w = torch.arange(0, camera.image_width, TILE_SIZE).to("cuda")
    hh, ww = torch.meshgrid(h, w, indexing='ij')
    coords = torch.stack([hh.flatten(), ww.flatten()],dim=1)

    def return_mask(coord):
        h, w = coord
        # check if the rectangle penetrate the tile
        over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
        over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
        in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
        #print(sum(in_mask), len(in_mask))
        #print(in_mask[:10])
        return in_mask

    in_masks = functorch.vmap(return_mask)(coords)

    i = 0
    dd_loss = 0
    nc_loss = 0

    #cutoff_avg = 0
    for h in range(0, camera.image_height, TILE_SIZE):
        for w in range(0, camera.image_width, TILE_SIZE):
            in_mask = in_masks[i]

            if not in_mask.sum() > 0:
                continue

            P = in_mask.sum()
            #tile_coord = pc.pix_coord[h][:,w].flatten(0,-2)
            tile_coord = pc.pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)

            sorted_depths = depths[in_mask]
            sorted_normal = normal[in_mask]

            with torch.no_grad():

                sorted_means2D = means2D[in_mask]
                sorted_cov2d = cov2d[in_mask] # P 2 2
                sorted_conic = sorted_cov2d.inverse() # inverse of variance
                sorted_opacity = opacity[in_mask]
                sorted_color = color[in_mask]
                dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
                
                gauss_weight = torch.exp(-0.5 * dx[:, :, 0]**2 * sorted_conic[:, 0, 0] + dx[:, :, 1]**2 * sorted_conic[:, 1, 1] + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1] + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0])
                #dist2d = dx[:, :, 0]**2 * sorted_conic[:, 0, 0] + dx[:, :, 1]**2 * sorted_conic[:, 1, 1] + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1] + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]
                #gauss_weight = torch.exp(-0.5*dist2d)
                
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                #print(alpha.numel(), (alpha>1./255).sum())
                #alpha = alpha[alpha>1./255]
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                #debug = max(T[:,:30][:,-1].max().item(), debug)
                #weights = (alpha*T).squeeze(-1).detach()
                weights = (alpha*T).squeeze(-1)
                inds = torch.randperm(weights.shape[0])
                inds = inds[:300]
                #ray_inds = torch.randperm(weights.shape[1])
                #ray_inds = ray_inds[:30]

                #weights_sel = weights[inds][:, ray_inds] # B P
                #weights_col = weights_sel.unsqueeze(-1)
                #weights_row = weights_sel.unsqueeze(-2)
                #weights_comb = weights_col * weights_row # B P P

            ## torch no grad ends

            #depths_sel = sorted_depths[ray_inds] # P
            d_col = sorted_depths.unsqueeze(1)
            d_row = sorted_depths.unsqueeze(0)
            d_comb = torch.abs(d_col - d_row) # P P

            dd_loss = dd_loss + torch.einsum('bn,bm,nm->', weights[inds], weights[inds], d_comb)

            #dd_loss += (weights_comb*d_comb).mean()

            #t_normal_d = normal_d[:,h][:,:,w].permute(1,2,0).reshape(-1,3)[inds]
            t_normal_d = normal_d[:,h:h+TILE_SIZE, w:w+TILE_SIZE].permute(1,2,0).reshape(-1,3)[inds]
            #t_normal_d = normal_d[:,h:h+TILE_SIZE, w:w+TILE_SIZE].permute(1,2,0).reshape(-1,3)

            #norm = (sorted_normal[ray_inds].unsqueeze(0) * t_normal_d.unsqueeze(1)).sum(-1)
            #norm = torch.einsum('ij,kj->ik', t_normal_d, sorted_normal[ray_inds])
            norm = torch.einsum('ij,kj->ik', t_normal_d, sorted_normal)
            nc_loss = nc_loss + (weights[inds]*(1-norm)).mean()
            i += 1


    #if nc_loss == 0:
    #    import pdb;pdb.set_trace()
    dd_loss = dd_loss / tile_num
    nc_loss = nc_loss / tile_num

    #dd_loss *= 0.01
    #nc_loss *= 1
    sorted_rets = [dd_loss, nc_loss]

    #print(cutoff_avg/tile_num)

    #save_image(render_color.permute(2,0,1), 'test/r.png') 
    #save_image(render_depth.permute(2,0,1).repeat(3,1,1), 'test/d.png') 
    #save_image(render_alpha.permute(2,0,1).repeat(3,1,1), 'test/o.png') 

    return sorted_rets
'''


'''
def torch_render3(pc, camera, means2D, cov2d, color, opacity, mean_view, bg, normal_d, normal):
    if (bg.cpu() == torch.tensor([1,1,1])).all():
        white_bkgd = True
    elif (bg.cpu() == torch.tensor([0,0,0])).all():
        white_bkgd = False
    else:
        assert False

    depths = mean_view[:,2]

    depths, index = torch.sort(depths)
    means2D = means2D[index]
    cov2d = cov2d[index]
    color = color[index]
    opacity = opacity[index]
    normal = normal[index]

    radii = get_radius(cov2d)
    rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
    
    TILE_SIZE = 16
    sorted_rets = []
    tile_num = (camera.image_height/TILE_SIZE)*(camera.image_width/TILE_SIZE)
    

    dd_loss = 0
    nc_loss = 0

    hs = random.sample(range(0, camera.image_height, TILE_SIZE), 10)
    ws = random.sample(range(0, camera.image_width, TILE_SIZE), 10)

    for h in hs:
        for w in ws:

            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 

            if not in_mask.sum() > 0:
                continue

            P = in_mask.sum()
            tile_coord = pc.pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
            sorted_depths = depths[in_mask]
            sorted_normal = normal[in_mask]

            with torch.no_grad():

                sorted_means2D = means2D[in_mask]
                sorted_cov2d = cov2d[in_mask] # P 2 2
                sorted_conic = sorted_cov2d.inverse() # inverse of variance
                sorted_opacity = opacity[in_mask]
                sorted_color = color[in_mask]
                dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2

                gauss_weight = torch.exp(-0.5 * dx[:, :, 0]**2 * sorted_conic[:, 0, 0] + dx[:, :, 1]**2 * sorted_conic[:, 1, 1] + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1] + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0])
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                weights = (alpha*T).squeeze(-1)
            ## torch no grad ends

            #normalized_depths = (torch.reciprocal(sorted_depths)-100)*(-100/9999.)
            normalized_depths = -torch.reciprocal(sorted_depths)
            #print(sorted_depths.max().item(), normalized_depths.max().item(), sorted_depths.min().item(), normalized_depths.min().item())

            d_col = sorted_depths.unsqueeze(1)
            d_row = sorted_depths.unsqueeze(0)
            d_comb = torch.abs(d_col - d_row) # P P

            dd_loss = dd_loss + torch.einsum('bn,bm,nm->', weights, weights, d_comb)



    dd_loss = dd_loss / tile_num
    sorted_rets = [dd_loss, 0]


    return sorted_rets
'''



def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def build_color(pc, means3D, shs, camera):
    rays_o = camera.camera_center
    rays_d = means3D - rays_o
    color = eval_sh(pc.active_sh_degree, shs.permute(0,2,1), rays_d)
    color = (color + 0.5).clip(min=0.0)
    return color

def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = torch.logical_and(p_view[..., 2] >= 0.01, p_view[..., 2] < 100)
    #print(p_view[...,2].min(), p_view[...,2].max())
    return p_proj, p_view, in_mask

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002). 
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]

'''
def torch_forward(camera, pc, bg, normal_d):
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    if pc.pix_coord is None:
        pc.pix_coord = torch.stack(torch.meshgrid(torch.arange(camera.image_width), torch.arange(camera.image_height), indexing='xy'), dim=-1).to('cuda')
    
    mean_ndc, mean_view, in_mask = projection_ndc(means3D, 
            viewmatrix=camera.world_view_transform, 
            projmatrix=camera.projection_matrix)
    mean_ndc = mean_ndc[in_mask]
    mean_view = mean_view[in_mask]
    #depths = mean_view[:,2]

    #####
    rotations_mat = build_rotation(rotations)
    min_scales = torch.argmin(scales, dim=1)
    indices = torch.arange(min_scales.shape[0])
    normal = rotations_mat[indices, :, min_scales]

    #view_dir = means3D - camera.camera_center
    #normal   = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]

    R_w2c = torch.tensor(camera.R.T).cuda().to(torch.float32)
    normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)
    normal = normal[in_mask]
    ####
    
    color = build_color(pc, means3D=means3D, shs=shs, camera=camera)
    
    cov3d = build_covariance_3d(scales, rotations)

    focal_x = fov2focal(camera.FoVx, camera.image_width)
    focal_y = fov2focal(camera.FoVy, camera.image_height)
        
    cov2d = build_covariance_2d(
        mean3d=means3D, 
        cov3d=cov3d, 
        viewmatrix=camera.world_view_transform,
        fov_x=camera.FoVx, 
        fov_y=camera.FoVy, 
        focal_x=focal_x, 
        focal_y=focal_y)


    mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
    mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
    means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

    cov2d = cov2d[in_mask]
    color = color[in_mask]
    opacity = opacity[in_mask]
    
    rets = torch_render3(
        pc=pc,
        camera = camera, 
        means2D=means2D,
        cov2d=cov2d,
        color=color,
        opacity=opacity, 
        mean_view=mean_view,
        bg=bg,
        normal_d=normal_d,
        normal=normal
    )

    return rets
'''



def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, return_depth = False, return_normal = False, no_torch=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    return_dict = {} 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python: cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_dict["render"] = rendered_image
    return_dict["viewspace_points"] = screenspace_points
    return_dict["visibility_filter"] = radii > 0
    return_dict["radii"] = radii

    torch_out = None

    if return_depth:
        with torch.no_grad():
            projvect1 = viewpoint_camera.world_view_transform[:,2][:3].detach()
            projvect2 = viewpoint_camera.world_view_transform[:,2][-1].detach()
            means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(dim=-1,keepdim=True) + projvect2
            means3D_depth = means3D_depth.repeat(1,3)
            render_depth, _ = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = means3D_depth,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

        density = torch.ones_like(means3D)
        render_opacity, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = density,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_opacity = render_opacity.mean(dim=0)
        render_opacity = render_opacity.clamp(1e-6, 1 - 1e-6)
        return_dict.update({'render_opacity': render_opacity})

        with torch.no_grad():
            render_depth = render_depth.mean(dim=0) / render_opacity

            return_dict.update({'render_depth': render_depth})


            ## depth2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normal_d = -depth_to_normals(render_depth[None][None], viewpoint_camera.intrinsics[None])[0]

        '''
        render_depth_np = render_depth.detach().cpu().numpy()
        dy, dx = np.gradient(render_depth_np, edge_order=2)

        fx = image_width / (2.*tanfovx)
        fy = image_height / (2.*tanfovy)

        A = np.dstack((render_depth_np / fx, np.zeros((image_height, image_width)), dx))
        B = np.dstack((np.zeros((image_height, image_width)), render_depth_np / fy, dy))
        normal_d = -np.cross(A,B)
        norm = np.sqrt(np.sum(normal_d**2, axis=2, keepdims=True))
        normal_d = np.divide(normal_d, norm, out=np.zeros_like(normal_d), where=norm != 0)
        normal_d = torch.tensor(normal_d).permute(2,0,1).cuda()
        '''

        '''
        if not no_torch:
            torch_out = torch_forward(viewpoint_camera, pc, bg_color, normal_d)
        '''

        return_dict.update({"render_normal_d": normal_d})
        #print(f'normal_d pos: {(normal_d[2,:,:]>0).sum()}')

        #return_dict.update({"render_normal_d2": normal_d2})

    
    if return_normal:
        rotations_mat = build_rotation(rotations)
        scales = pc.get_scaling
        min_scales = torch.argmin(scales, dim=1)
        indices = torch.arange(min_scales.shape[0])
        normal = rotations_mat[indices, :, min_scales]

        #view_dir = means3D - viewpoint_camera.camera_center
        #normal   = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]

        #R_w2c = torch.tensor(viewpoint_camera.R.T).cuda().to(torch.float32)
        #normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)

        with torch.no_grad():
            render_normal, _ = rasterizer(
                means3D = means3D.detach(),
                means2D = means2D.detach(),
                shs = None,
                colors_precomp = normal,
                opacities = opacity.detach(),
                scales = scales.detach(),
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
            render_normal = F.normalize(render_normal, dim = 0)

            return_dict["render_normal"] = render_normal

        # w2c
        #view_dir = means3D - viewpoint_camera.camera_center
        #normal   = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
        R_w2c = torch.tensor(viewpoint_camera.R.T).cuda().to(torch.float32)
        normal_cam = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)
        render_normal_cam, _ = rasterizer(
            means3D = means3D.detach(),
            means2D = means2D.detach(),
            shs = None,
            colors_precomp = normal_cam,
            opacities = opacity.detach(),
            scales = scales.detach(),
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_normal_cam = F.normalize(render_normal_cam, dim = 0)
        #print(f'normal pos: {(render_normal_cam[2,:,:]>0).sum()}')
        return_dict["render_normal_cam"] = render_normal_cam


    return return_dict, torch_out

