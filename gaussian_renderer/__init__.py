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
#from diff_gaussian_rasterization2 import GaussianRasterizationSettings as GaussianRasterizationSettings2, GaussianRasterizer as GaussianRasterizer2
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
from utils.graphics_utils import fov2focal
import torch.nn.functional as F
#from kornia.geometry import depth_to_normals
from torchvision.utils import save_image
from utils.depth_utils import depth_to_normal
from utils.get_stats import get_effective_rank


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



def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, return_depth = False, return_normal = False):
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

    subpixel_offset = torch.zeros((image_height, image_width, 2), dtype=torch.float32, device="cuda")

    # code borrowed from gaussian opacity fields
    # raster_settings_2 = GaussianRasterizationSettings2(
    #     image_height=image_height,
    #     image_width=image_width,
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     kernel_size=0.,
    #     subpixel_offset=subpixel_offset,
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=viewpoint_camera.world_view_transform,
    #     projmatrix=viewpoint_camera.full_proj_transform,
    #     sh_degree=pc.active_sh_degree,
    #     campos=viewpoint_camera.camera_center,
    #     prefiltered=False,
    #     debug=pipe.debug
    # )

    #rasterizer_2 = GaussianRasterizer2(raster_settings=raster_settings_2)

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


    '''
    rendered_image_2,_ = rasterizer_2(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=None)

    return_dict['render_2'] = rendered_image_2
    '''
    return_dict['render_2'] = None

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
            ## testing
            #normal_d3, _ = depth_to_normal(viewpoint_camera, render_depth[None])
            #normal_d3 = normal_d3.permute(2,0,1)

            #import pdb;pdb.set_trace()

        #return_dict.update({"render_normal_d": normal_d})
        #return_dict.update({"render_normal_d3": normal_d3})

    '''
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

        render_normal, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = normal,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_normal = F.normalize(render_normal, dim = 0)

        return_dict["render_normal"] = render_normal

        with torch.no_grad():
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
    '''


    return return_dict


def render_erank(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, return_depth = False, return_normal = False):
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

    subpixel_offset = torch.zeros((image_height, image_width, 2), dtype=torch.float32, device="cuda")

    # code borrowed from gaussian opacity fields
    raster_settings_2 = GaussianRasterizationSettings2(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=0.,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer_2 = GaussianRasterizer2(raster_settings=raster_settings_2)

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


    '''
    rendered_image_2,_ = rasterizer_2(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=None)

    return_dict['render_2'] = rendered_image_2
    '''
    return_dict['render_2'] = None

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

    render_depth = render_depth.mean(dim=0) / render_opacity

    return_dict.update({'render_depth': render_depth})

    erank = get_effective_rank(scales)
    #thres=1.005
    thres = 1.025
    print(len(erank))
    print((erank<thres).sum())
    #erank = torch.where(erank<1.005, 1., 0.)
    erank = torch.where(erank<thres, 1., 0.)
    erank = erank.unsqueeze(1).repeat(1,3)
    mask = erank.any(dim=1)
    erank[mask] = torch.tensor([1,0,0.],dtype=torch.float32).cuda()
    #inv_erank = -erank + 4
    render_erank, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = erank,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    print(render_erank.max(), render_erank.min())
    red_channel = render_erank[0, :, :]  # Red channel is the first channel
    red_mask = red_channel > 0.01
    red_mask = red_mask.unsqueeze(0).repeat(3,1,1)
    # Find the locations where the red channel value is larger than 0.9
    render_mixed = rendered_image.clone()
    render_mixed[red_mask] = (render_erank[red_mask]*0.7+rendered_image[red_mask]*0.3)
    #render_mixed = torch.clamp(0.6*render_erank + 0.8*rendered_image, 0,1.)
    #render_erank = torch.clamp((render_erank - 1) / 2,0)
    return_dict.update({'render_erank': render_erank})
    return_dict.update({'render_mixed': render_mixed})

    rotations_mat = build_rotation(rotations)
    scales = pc.get_scaling
    min_scales = torch.argmin(scales, dim=1)
    indices = torch.arange(min_scales.shape[0])
    normal = rotations_mat[indices, :, min_scales]

    #view_dir = means3D - viewpoint_camera.camera_center
    #normal   = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]

    #R_w2c = torch.tensor(viewpoint_camera.R.T).cuda().to(torch.float32)
    #normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)

    render_normal, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = normal,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    render_normal = F.normalize(render_normal, dim = 0)

    view_dir = means3D - viewpoint_camera.camera_center
    normal   = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
    render_normal_2, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = normal,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    render_normal_2 = F.normalize(render_normal_2, dim = 0)

    return_dict["render_normal"] = render_normal
    return_dict["render_normal_2"] = render_normal_2

    normal_d3, _ = depth_to_normal(viewpoint_camera, render_depth[None])
    normal_d3 = normal_d3.permute(2,0,1)
    return_dict["render_normal_d"] = normal_d3


    return return_dict

