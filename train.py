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

import os
import io
import wandb
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, vis_depth, build_rotation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.depth_utils import depth_to_normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.get_stats import get_effective_rank, get_volume, get_ordered_scale_multiple
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from datetime import datetime as dt
from PIL import Image
import numpy as np



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad


def get_histograms(log_dict, iteration, scene, erank, volume, opacity, ordered_scale_multiple):
    erank_np = erank.detach().cpu().numpy()
    volume_np = volume.detach().cpu().numpy()
    opacity_np = opacity.detach().cpu().numpy()
    ordered_scale_multiple_np = ordered_scale_multiple.detach().cpu().numpy()

    # erank histogram
    plt.clf()
    mean_erank = torch.mean(erank).item()
    #print(f'iteration: {iteration}  erank: {mean_erank}')
    fig, ax1 = plt.subplots(figsize=(8,6))

    plt.ylim(0, 30000)
    plt.xticks([1, 1.5, 2.0, 2.5, 3.0])  # Specify x-axis tick marks
    plt.yticks([5000,15000,25000])
    ax1.tick_params(axis='both', which='major', labelsize=24)

    cmap=plt.cm.Greens
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0., vmax=3.0)
    #ax1.hist(erank_np, bins=50, range=(1.,3.), color='green', alpha=0.5)  # You can adjust the number of bins to your preference
    n,bins,patches=ax1.hist(erank_np, bins=50, range=(1.,3.), alpha=0.5)  # You can adjust the number of bins to your preference
    for patch, value in zip(patches, bins):
        color = cmap(norm(value))
        patch.set_facecolor(color)

    ax1.text(0.65, 0.9, f'iteration: {iteration}', fontsize=12,transform=plt.gca().transAxes)
    ax1.text(0.65, 0.77, f'total: {len(scene.gaussians.get_xyz)}', fontsize=12, transform=plt.gca().transAxes)
    ax1.text(0.65, 0.83, f'mean: {mean_erank:.3f}', fontsize=12, transform=plt.gca().transAxes)

    #ax1.set_xlabel('effective rank', fontsize=24)
    #ax1.set_ylabel('count', fontsize=24)
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.9)


    erank_bin_edges = np.histogram_bin_edges(erank_np, bins=50, range=(1.,3.))
    # avg_volume_erank
    average_volume_erank = []
    average_opacity_erank = []
    erank_num = []
    for i in range(50):
        if i == 49:
            erank_ind = np.where((erank_np >= erank_bin_edges[i]) & (erank_np <= erank_bin_edges[i+1]))[0]
        else:
            erank_ind = np.where((erank_np >= erank_bin_edges[i]) & (erank_np < erank_bin_edges[i+1]))[0]
        erank_num.append(len(erank_ind))
        volume_in_erank_bin = [volume_np[idx] for idx in erank_ind]
        opacity_in_erank_bin = [opacity_np[idx] for idx in erank_ind]
        _average_volume_erank = np.mean(volume_in_erank_bin)
        _average_opacity_erank = np.mean(opacity_in_erank_bin)
        average_volume_erank.append(_average_volume_erank)
        average_opacity_erank.append(_average_opacity_erank)
    average_volume_erank = np.array(average_volume_erank)
    average_opacity_erank = np.array(average_opacity_erank)
    average_volume_erank[np.isnan(average_volume_erank)] = 0
    average_opacity_erank[np.isnan(average_opacity_erank)] = 0

    smoothed_volume_erank = []
    smoothed_opacity_erank = []
    for i in range(50):
        if i == 0:
            smoothed_volume_erank.append(average_volume_erank[i])
            smoothed_opacity_erank.append(average_opacity_erank[i])
        elif i ==1:
            if erank_num[i] + erank_num[i-1] != 0:
                sv = (average_volume_erank[i]*erank_num[i] + average_volume_erank[i-1]*erank_num[i-1]) / (erank_num[i] + erank_num[i-1])
                so = (average_opacity_erank[i]*erank_num[i] + average_opacity_erank[i-1]*erank_num[i-1]) / (erank_num[i] + erank_num[i-1])
            else:
                sv = so = 0
            smoothed_volume_erank.append(sv)
            smoothed_opacity_erank.append(so)

        else:
            if erank_num[i] + erank_num[i-1] + erank_num[i-2] != 0:
                sv = (average_volume_erank[i]*erank_num[i] + average_volume_erank[i-1]*erank_num[i-1] + average_volume_erank[i-2]*erank_num[i-2]) / (erank_num[i] + erank_num[i-1] + erank_num[i-2])
                so = (average_opacity_erank[i]*erank_num[i] + average_opacity_erank[i-1]*erank_num[i-1] + average_opacity_erank[i-2]*erank_num[i-2]) / (erank_num[i] + erank_num[i-1] + erank_num[i-2])
            else:
                sv = so = 0
            smoothed_volume_erank.append(sv)
            smoothed_opacity_erank.append(so)



    os.makedirs(os.path.join(scene.model_path, 'stats', 'eranks'), exist_ok=True)
    plt.savefig(os.path.join(scene.model_path, 'stats', 'eranks', f'{iteration:05}.png'))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    erank_array = np.array(Image.open(buf))
    # wandb
    erank_histogram = torch.tensor(erank_array[:,:,:3]/255.).permute(2,0,1)
    log_dict['erank_histogram'] = wandb.Image(erank_histogram)


    ########################
    # opacity histogram
    ########################
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(8,6))
    plt.xticks([0.,0.2,0.4,0.6,0.8,1.0])  # Specify x-axis tick marks
    plt.ylim(0, 30000)

    ax1.hist(opacity_np, bins=50, range=(0.,1.), color='green', alpha=0.5)  # You can adjust the number of bins to your preference
    plt.title(f"{scene.model_path.split('/')[-2]}/{scene.model_path.split('/')[-1]}", loc='center')

    ax1.text(0.65, 0.9, f'iter: {iteration}', fontsize=12,transform=plt.gca().transAxes)
    ax1.text(0.65, 0.83, f'# gaussians: {len(scene.gaussians.get_xyz)}', fontsize=12, transform=plt.gca().transAxes)
    ax1.text(0.65, 0.77, f'mean_erank: {mean_erank:.3f}', fontsize=12, transform=plt.gca().transAxes)

    ax1.set_xlabel('opacity')
    ax1.set_ylabel('opacity hist', color='green')

    opacity_bin_edges = np.histogram_bin_edges(opacity_np, bins=50, range=(0.,1.))
    average_volume_opacity = []
    average_erank_opacity = []
    opacity_num = []

    for i in range(50):
        if i == 49:
            opacity_ind = np.where((opacity_np >= opacity_bin_edges[i]) & (opacity_np <= opacity_bin_edges[i+1]))[0]
        else:
            opacity_ind = np.where((opacity_np >= opacity_bin_edges[i]) & (opacity_np < opacity_bin_edges[i+1]))[0]
        opacity_num.append(len(opacity_ind))
        volume_in_opacity_bin = [volume_np[idx] for idx in opacity_ind]
        erank_in_opacity_bin = [erank_np[idx] for idx in opacity_ind]
        _average_volume_opacity = np.mean(volume_in_opacity_bin)
        _average_erank_opacity = np.mean(erank_in_opacity_bin)
        average_volume_opacity.append(_average_volume_opacity)
        average_erank_opacity.append(_average_erank_opacity)
    average_volume_opacity = np.array(average_volume_opacity)
    average_erank_opacity = np.array(average_erank_opacity)
    average_volume_opacity[np.isnan(average_volume_opacity)] = 0
    average_erank_opacity[np.isnan(average_erank_opacity)] = 0

    smoothed_volume_opacity = []
    smoothed_erank_opacity = []
    for i in range(50):
        if i == 0:
            smoothed_volume_opacity.append(average_volume_opacity[i])
            smoothed_erank_opacity.append(average_erank_opacity[i])
        elif i == 1:
            if opacity_num[i] + opacity_num[i-1] != 0:
                svo = (average_volume_opacity[i]*opacity_num[i] + average_volume_opacity[i-1]*opacity_num[i-1]) / (opacity_num[i]+opacity_num[i-1])
                seo = (average_erank_opacity[i]*opacity_num[i] + average_erank_opacity[i-1]*opacity_num[i-1]) / (opacity_num[i]+opacity_num[i-1])
            else:
                svo = seo = 0
            smoothed_volume_opacity.append(svo)
            smoothed_erank_opacity.append(seo)
        else:
            if opacity_num[i] + opacity_num[i-1] + opacity_num[i-2] != 0:
                svo = (average_volume_opacity[i]*opacity_num[i] + average_volume_opacity[i-1]*opacity_num[i-1] + average_volume_opacity[i-2]*opacity_num[i-2]) / (opacity_num[i]+opacity_num[i-1]+opacity_num[i-2])
                seo = (average_erank_opacity[i]*opacity_num[i] + average_erank_opacity[i-1]*opacity_num[i-1] + average_erank_opacity[i-2]*opacity_num[i-2]) / (opacity_num[i]+opacity_num[i-1]+opacity_num[i-2])
            else:
                svo = seo = 0
            smoothed_volume_opacity.append(svo)
            smoothed_erank_opacity.append(seo)

    ax2 = ax1.twinx()
    ax2.set_ylabel('volume', color='red')
    ax2.set_ylim([0, 1e-6])

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('erank', color='purple')
    ax3.set_ylim([1,3.])

    x_values = []
    for i in range(len(opacity_bin_edges)-1):
        x_values.append((opacity_bin_edges[i]+opacity_bin_edges[i+1])/2.)

    ax2.plot(x_values, smoothed_volume_opacity, marker='o', linestyle='--', color='red', alpha=0.5)
    ax3.plot(x_values, smoothed_erank_opacity, marker='o', linestyle='-', color='purple',alpha=0.5)

    plt.subplots_adjust(right=0.8)

    os.makedirs(os.path.join(scene.model_path, 'stats', 'opacity'), exist_ok=True)
    plt.savefig(os.path.join(scene.model_path, 'stats', 'opacity', f'{iteration:05}.png'))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    opacity_histogram = np.array(Image.open(buf))

    #wandb
    opacity_histogram = torch.tensor(opacity_histogram[:,:,:3]/255.).permute(2,0,1)
    log_dict['opacity_histogram'] = wandb.Image(opacity_histogram)


    ########################
    # ordered_scale_multiple
    ########################
    plt.clf()
    fig, axs = plt.subplots(1,2,figsize=(12,5))

    x_mult = np.clip(ordered_scale_multiple_np[:,0], a_min=0, a_max=100)
    y_mult = np.clip(ordered_scale_multiple_np[:,1], a_min=0, a_max=100)
    mappable = axs[0].hist2d(x_mult, y_mult, bins=50, cmap='BuPu', range=[[1,100],[1,100]])


    # Add labels and title
    axs[0].set_xlabel('1st / 3rd')
    axs[0].set_ylabel('2nd / 3rd')
    axs[0].set_title('scale multiplier w.r.t. 3rd scale')

    axs[0].text(0.2, 0.9, f'iter: {iteration}', fontsize=8,transform=plt.gca().transAxes)
    axs[0].text(0.2, 0.83, f'# gaussians: {len(scene.gaussians.get_xyz)}', fontsize=8, transform=plt.gca().transAxes)
    axs[0].text(0.2, 0.77, f'mean_erank: {mean_erank:.3f}', fontsize=8, transform=plt.gca().transAxes)

    # Add color bar
    fig.colorbar(mappable[3], ax=axs[0])

    mult_12 = np.clip(ordered_scale_multiple_np[:,0] / ordered_scale_multiple_np[:,1], a_min=0, a_max=100)
    axs[1].hist(mult_12, bins=50, range=[1,100])
    axs[1].set_xlabel('1st / 2nd')
    axs[1].set_title('scale multiplier w.r.t. 2nd scale')

    os.makedirs(os.path.join(scene.model_path, 'stats', 'scale'), exist_ok=True)
    plt.savefig(os.path.join(scene.model_path, 'stats', 'scale', f'{iteration:05}.png'))
    plt.close()


    return log_dict


def training(dataset, opt, pipe, exp_name, test_freq, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    data_name = os.path.basename(dataset.source_path)

    # Render
    if debug_from == 0:
        pipe.debug = True


    out_path = f'output/{data_name}/' + exp_name

    dataset.model_path = out_path

    prepare_output_and_logger(dataset, opt, data_name, debug=pipe.debug)

    first_iter = 0

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        scale = gaussians.get_scaling
        erank = get_effective_rank(scale)
        volume = get_volume(scale)
        ordered_scale_multiple, ordered_scale = get_ordered_scale_multiple(scale)
        opacity = scene.gaussians.get_opacity


        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_normal=True, return_depth=True)

        image, rendering_2, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_2"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        recon_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))


        erank_loss = 0
        thin_loss = 0

        if iteration >= opt.erank_from_iter and iteration <= opt.erank_end_iter:
            # erank loss
            erank_loss = opt.erank_lambda * torch.clamp(-torch.log(erank-1+1e-7), 0).mean()

            thin_loss = opt.thin_lambda*ordered_scale[:,2].mean()

        loss = recon_loss + erank_loss + thin_loss

        image_mask = viewpoint_cam.image_mask
        if image_mask:
            o = render_pkg['render_opacity']
            opacity_mask_loss = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
            loss += opacity_mask_loss 

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if not pipe.debug:
                training_report_wandb(wandb, iteration, recon_loss, erank_loss, thin_loss, loss, erank, volume, opacity, ordered_scale_multiple, iter_start.elapsed_time(iter_end), test_freq, scene, render, (pipe, background), first_iter)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if opt.no_new_densification:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    else:
                        gaussians.densify_and_prune_v2(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    return out_path

def prepare_output_and_logger(args, opt, data_name, debug=False):    
    if not debug:
        print('init wandb')
        os.environ["WANDB__SERVICE_WAIT"]="300"
        project_name = f"{data_name}_erank"
        if args.fewshot:
            project_name = '[fewshot]' + project_name
        wandb.init(project=project_name, name=args.model_path, config=vars(opt))

    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    return

def training_report_wandb(wandb, iteration, recon_loss, erank_loss, thin_loss, loss, erank, volume, opacity, ordered_scale_multiple, elapsed, test_freq, scene : Scene, renderFunc, renderArgs, first_iter):

    log_dict = {"total_loss": loss, "recon_loss": recon_loss, "erank_loss": erank_loss, "thin_loss": thin_loss, "elapsed": elapsed, "erank_avg": torch.mean(erank).item()}
    log_dict['total_points'] = scene.gaussians.get_xyz.shape[0]
    if iteration % 10 == 0:
        wandb.log(log_dict, step=iteration)

    # Report test and samples of training set
    if iteration % test_freq == 0 or iteration == first_iter:
        torch.cuda.empty_cache()

        # save and return histograms
        log_dict = get_histograms(log_dict, iteration, scene, erank, volume, opacity, ordered_scale_multiple)

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                images = []
                gt_images = []
                normal_d_images = []
                depth_images = []
                opacity_images = []
                normal_images_gof = []
                for idx, viewpoint in enumerate(config['cameras']):
                    torch.cuda.empty_cache()
                    render_pkg = renderFunc(viewpoint, scene.gaussians, return_depth=True, return_normal=True, *renderArgs)
                    rendering_2 = render_pkg['render_2']

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    if idx < 5:
                        opacity = render_pkg['render_opacity']
                        opacity = opacity.cpu()

                        gt_images.append(gt_image[None].cpu())
                        images.append(image[None].cpu())
                        opacity_images.append(opacity.repeat(3,1,1)[None])

                gt_images = torch.cat(gt_images, dim=0)
                images = torch.cat(images, dim=0)
                opacity_images = torch.cat(opacity_images, dim=0)

                gt_images = make_grid(gt_images, nrow=5)
                images = make_grid(images, nrow=5)
                opacity_images = make_grid(opacity_images, nrow=5)

                log_dict[config['name'] + "_render"] = wandb.Image(images)
                log_dict[config['name'] + "_opacity_render"] = wandb.Image(opacity_images)

                if iteration == first_iter:
                    log_dict[config['name'] + "_gt"] = wandb.Image(gt_images)


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                log_dict[config['name'] + '_recon_loss'] = l1_test
                log_dict[config['name'] + '_psnr_test'] = psnr_test

        wandb.log(log_dict, step=iteration)
        torch.cuda.empty_cache()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_freq", type=int, default=2000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--exp_name", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    exp_name = training(lp.extract(args), op.extract(args), pp.extract(args), args.exp_name, args.test_freq, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")

    script = f'python extract_mesh_tsdf.py -m {exp_name} --iteration 30000'
    print(script)
    os.system(script)

    #optional: evaluation
    script = f'python evaluate_dtu_mesh.py -m {exp_name}'
    print(script)
    os.system(script) 
    
