import os
import argparse
import time
import random
import numpy as np
import torch
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import collections.abc
container_abcs = collections.abc
from itertools import repeat
import scipy.io as sio
import numpy.linalg as LA
import scipy.spatial.distance as scipy_spatial_dist
from scipy.interpolate import griddata

import argparse
import os
import random
import numpy as np
import torch
from numpy import linalg as LA

def cosine_distance(x, y, semi_sphere=False):
    dist_cosine = scipy_spatial_dist.cdist(x, y, 'cosine')
    dist_cosine *= -1.0
    dist_cosine += 1.0

    if semi_sphere:
        dist_cosine = np.abs(dist_cosine)
    
    dist_cosine_arc = np.arccos(dist_cosine)
    return dist_cosine_arc

def cartesian_to_spherical(xyz):
    num_points = len(xyz)
    angles = np.zeros((num_points, 2))
    angles[:, 1] = np.arcsin(xyz[:, 1])
    inner = xyz[:, 0] / np.cos(angles[:, 1])
    inner = np.clip(inner, a_min=-1.0, a_max=1.0)
    angles[:, 0] = np.arcsin(inner)
    return angles

def spherical_to_cartesian(angles):
    num_points = len(angles)
    xyz = np.zeros((num_points, 3))
    xyz[:, 0] = np.sin(angles[:, 0]) * np.cos(angles[:, 1])
    xyz[:, 1] = np.sin(angles[:, 1])
    xyz[:, 2] = np.cos(angles[:, 0]) * np.cos(angles[:, 1])
    return xyz

def gold_spiral_sampling(v, alpha, num_pts):
    v1 = orthogonal_vector(v)
    v2 = np.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = np.arange(num_pts) + 0.5
    phi = np.arccos(1 + (np.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    return (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T

def orthogonal_vector(v):
    x, y, z = v
    orthogonal = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    orthogonal /= LA.norm(orthogonal)
    return orthogonal

def to_pixel_coordinates(vpts, focal_length=1.0, height=480, width=640):
    x = vpts[:, 0] / vpts[:, 2] * focal_length * max(height, width) / 2.0 + width // 2
    y = -vpts[:, 1] / vpts[:, 2] * focal_length * max(height, width) / 2.0 + height // 2
    return y, x

def hough_transformation(rows, cols, theta_res, rho_res):
    theta = np.linspace(0, 180.0, int(np.ceil(180.0 / theta_res) + 1.0)) + 0.5
    theta = theta[:-1]
    D = np.sqrt((rows // 2 + 0.5) ** 2 + (cols // 2 + 0.5) ** 2)
    rho = np.arange(-D, D + rho_res, rho_res)

    w = np.size(theta)
    h = np.size(rho)
    cos_values = np.cos(theta * np.pi / 180.0)
    sin_values = np.sin(theta * np.pi / 180.0)
    sin_cos = np.concatenate((sin_values[None, :], cos_values[None, :]), axis=0)

    coords_r, coords_w = np.ones((rows, cols)).nonzero()
    coords = np.concatenate((coords_r[:, None], coords_w[:, None]), axis=1).astype(np.float32)
    coords += 0.5

    coords[:, 0] = -1.0 * (coords[:, 0] - rows // 2)
    coords[:, 1] = coords[:, 1] - cols // 2

    vote_map = (coords @ sin_cos).astype(np.float32)

    mapping = []
    for i in range(rows * cols):
        for j in range(w):
            rho_val = vote_map[i, j]
            dis = (rho - rho_val)
            argsort = np.argsort(np.abs(dis))
            mapping.append(np.array([i, argsort[0] * w + j, dis[argsort[0]]]))

    return np.vstack(mapping).astype(np.float32), rho.astype(np.float32), theta.astype(np.float32)

def compute_normals(mapping_ht, img_height, img_width, ht_height, ht_width, rhos, thetas, focal_length):
    ht_normals = np.zeros((ht_height, ht_width, 3))
    ht_bin_invalid = []

    for i in range(ht_width):
        for j in range(ht_height):
            ht_inds = mapping_ht[:, 1] == j * ht_width + i
            if not ht_inds.any():
                continue
            
            rho = rhos[j]
            theta = thetas[i]

            if theta == 0.0: theta += 1e-16
            if rho == 0.0: rho += 1e-16

            valid_points = []

            x_boundary = (rho - (img_height // 2 - 0.5) * np.sin(theta)) / np.cos(theta)
            if -img_width // 2 <= x_boundary <= img_width // 2:
                valid_points.append([x_boundary, img_height // 2 - 0.5])

            y_boundary = (rho - (img_width // 2 - 0.5) * np.cos(theta)) / np.sin(theta)
            if -img_height // 2 <= y_boundary <= img_height // 2:
                valid_points.append([img_width // 2 - 0.5, y_boundary])

            if len(valid_points) < 2:
                ht_bin_invalid.append([j, i])
                continue

            valid_points = np.array(valid_points)

            if len(valid_points) > 2:
                dist01 = np.linalg.norm(valid_points[0] - valid_points[1])
                dist02 = np.linalg.norm(valid_points[0] - valid_points[2])
                dist12 = np.linalg.norm(valid_points[1] - valid_points[2])
                if dist01 == max(dist01, dist02, dist12):
                    valid_points = valid_points[:2]
                elif dist02 == max(dist01, dist02, dist12):
                    valid_points = np.delete(valid_points, 1, axis=0)
                else:
                    valid_points = valid_points[1:]

            points = np.concatenate([valid_points, focal_length * np.ones((len(valid_points), 1))], axis=1)
            points[:, 0] /= max(img_height, img_width) // 2
            points[:, 1] /= max(img_height, img_width) // 2

            normal_vector = np.cross(points[0], points[1])
            normal_vector /= max(np.linalg.norm(normal_vector), 1e-16)
            if normal_vector[2] < 0.0: 
                normal_vector *= -1
            ht_normals[j, i] = normal_vector

    return ht_normals.astype(np.float32)

def compute_sphere_points(ht_normals, num_samples):
    alphas_ = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=num_samples + 1, dtype=np.float64)
    alphas_ = alphas_[:-1]

    betas_ = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=num_samples + 1, dtype=np.float64)
    betas_ = betas_[:-1]

    all_hw_xyz = []

    ht_height, ht_width, _ = ht_normals.shape
    for i in range(ht_height * ht_width):
        cur_xyz = []
        norm_vector = ht_normals[i // ht_width, i % ht_width]
        if not np.sum(np.abs(norm_vector)) > 0.0: 
            continue

        if np.abs(norm_vector[1]) <= 1e-16:
            alpha_0 = np.arctan(- norm_vector[2] / (max(norm_vector[0], 1e-16) if norm_vector[0] >= 0.0 else min(norm_vector[0], -1e-16)))
            angles_alphas_0_betas = np.concatenate((np.ones((num_samples, 1)) * alpha_0, betas_[:, None]), axis=1)
            xyz_alphas_0_betas = spherical_to_cartesian(angles_alphas_0_betas)
            cur_xyz.append(xyz_alphas_0_betas)
        else:
            betas_f = np.arctan((-norm_vector[0] * np.sin(alphas_) - norm_vector[2] * np.cos(alphas_)) / (max(norm_vector[1], 1e-16) if norm_vector[1] >= 0.0 else min(norm_vector[1], -1e-16)))
            angles_alphas_betas_f = np.concatenate((alphas_[:, None], betas_f[:, None]), axis=1)
            xyz_alphas_betas_f = spherical_to_cartesian(angles_alphas_betas_f)
            cur_xyz.append(xyz_alphas_betas_f)

            norm_xz = np.sqrt(norm_vector[0] ** 2 + norm_vector[2] ** 2)
            norm_xz = max(norm_xz, 1e-16)
            t = -norm_vector[1] * np.tan(betas_) / norm_xz
            t_inds = np.logical_and(t >= -1, t <= 1)
            if t_inds.sum() == 0.0: 
                continue
            t = t[t_inds]
            betas = betas_[t_inds]

            alphas_phi = np.arccos(t)
            x_z = norm_vector[0] / max(norm_vector[2], 1e-16) if norm_vector[2] >= 0 else norm_vector[0] / min(norm_vector[2], -1e-16)
            phi = np.arctan(x_z)

            alphas_f1 = alphas_phi + phi
            alphas_f2 = -1.0 * alphas_phi + phi
            inds1 = np.logical_and(alphas_f1 >= -np.pi / 2, alphas_f1 < np.pi / 2)
            alphas_f = np.where(inds1, alphas_f1, alphas_f2)
            inds2 = np.logical_and(alphas_f >= -np.pi / 2, alphas_f < np.pi / 2)
            alphas_f = alphas_f[inds2]
            betas = betas[inds2]

            angles_alphas_f_betas = np.concatenate((alphas_f[:, None], betas[:, None]), axis=1)
            xyz_alphas_f_betas = spherical_to_cartesian(angles_alphas_f_betas)

            z_inds = xyz_alphas_f_betas[:, 2] >= 0.0
            xyz_alphas_f_betas = xyz_alphas_f_betas[z_inds]
            cur_xyz.append(xyz_alphas_f_betas)

        cur_xyz = np.vstack(cur_xyz)

        cur_hw = np.zeros((len(cur_xyz), 1), dtype=np.float64)
        cur_hw.fill(float(i))
        cur_hw_xyz = np.concatenate([cur_hw, cur_xyz], axis=-1)
        all_hw_xyz.append(cur_hw_xyz)

    all_hw_xyz = np.vstack(all_hw_xyz)
    return all_hw_xyz.astype(np.float32)

def find_nearest_neighbors_sphere(all_hw_xyz, xyz):
    if torch.cuda.is_available():
        all_hw_xyz = torch.from_numpy(all_hw_xyz).float().cuda()
        xyz = torch.from_numpy(xyz).float().cuda()
    else:
        all_hw_xyz = torch.from_numpy(all_hw_xyz).float()
        xyz = torch.from_numpy(xyz).float()

    all_hw = all_hw_xyz[:, 0]
    all_xyz = all_hw_xyz[:, 1:]
    all_xyz /= torch.norm(all_xyz, dim=1, keepdim=True)
    mapping = []

    max_hw = all_hw.max().long().item()
    for hw_ind in range(max_hw):
        hw_inds = all_hw == hw_ind
        if not hw_inds.any():
            continue
        clip_xyz = all_xyz[hw_inds]

        dist_cos = xyz @ clip_xyz.t()
        dist_cos = dist_cos.abs()
        max_values, max_inds = torch.max(dist_cos, dim=0)

        unique_inds = torch.unique(max_inds)
        clip_mapping = clip_xyz.new_zeros(len(unique_inds), 3).float()
        clip_mapping[:, 0] = hw_ind
        clip_mapping[:, 1] = unique_inds
        for ind, unique_ind in enumerate(unique_inds):
            max_max_value = max_values[max_inds == unique_ind].max()
            clip_mapping[ind, 2] = max_max_value

        mapping.append(clip_mapping.cpu().numpy())

    mapping = np.concatenate(mapping, axis=0)
    return mapping

def main(options):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rows = options.rows
    cols = options.cols
    theta_res = options.theta_res
    rho_res = options.rho_res
    num_samples = options.num_samples
    num_points = options.num_points
    focal_length = options.focal_length
    fibonacci_xyz = gold_spiral_sampling(np.array([0, 0, 1]), alpha=np.pi / 2, num_points=num_points)

    mapping_ht, rho, theta = hough_transformation(rows, cols, theta_res, rho_res)
    h = len(rho)
    w = len(theta)
    ht_npz_name = f"ht_{rows:d}_{cols:d}_{h:d}_{w:d}.npz"
    np.savez(os.path.join(save_dir, ht_npz_name),
             ht_mapping=mapping_ht,
             rho=rho,
             theta=theta,
             rows=rows,
             cols=cols,
             h=h,
             w=w,
             theta_res=theta_res,
             rho_res=rho_res)

    ht_normals = compute_normals(mapping_ht, rows, cols, h, w, rho, theta, focal_length)

    all_hw_xyz = compute_sphere_points(ht_normals, num_samples=num_samples)

    sphere_neighbors = find_nearest_neighbors_sphere(all_hw_xyz, fibonacci_xyz)

    sphere_neighbors_npz_name = f"sphere_neighbors_{h:d}_{w:d}_{num_points:d}.npz"
    np.savez(os.path.join(save_dir, sphere_neighbors_npz_name),
             h=h, w=w,
             num_points=num_points,
             num_samples=num_samples,
             xyz=fibonacci_xyz,
             focal_length=focal_length,
             sphere_neighbors_weight=sphere_neighbors)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', default='/parameterization/', help='path to save parameterizations')
    parser.add_argument('--focal_length', type=float, default=1.0, help='focal length, set to 1.0 if unknown')
    parser.add_argument('--rows', type=int, default=256, help='rows - image height')
    parser.add_argument('--cols', type=int, default=256, help='cols - image width')
    parser.add_argument('--theta_res', type=float, default=1.0, help='theta_res - hyperparameter for HT')
    parser.add_argument('--rho_res', type=float, default=1.0, help='rho_res - hyperparameter for HT')
    parser.add_argument('--num_samples', type=int, default=180, help='num_samples - number of angles for Gaussian sphere')
    parser.add_argument('--num_points', type=int, default=32768, help='num_points - number of sampled spherical points')
    options = parser.parse_args()
    print('options', options)
    main(options)
