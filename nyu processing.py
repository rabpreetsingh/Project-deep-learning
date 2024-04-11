# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:05:27 2024

@author: Rabpreet
"""

import glob
import os
import csv
import copy
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import argparse
import skimage.io as sio
import h5py

def rgb_to_gray(rgb_image):
    return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])

def to_pixel_coordinates(vpts, focal_length=1.0, image_height=480, image_width=640):
    x = vpts[:, 0] / vpts[:, 2] * focal_length * max(image_height, image_width) / 2.0 + image_width // 2
    y = -vpts[:, 1] / vpts[:, 2] * focal_length * max(image_height, image_width) / 2.0 + image_height // 2
    return y, x

class NYUVanishingPointsDataset:

    def __init__(self, data_directory="C:/Users/Rabpreet/Desktop/nyu_vp-master/nyu_vp-master/data", split='all', keep_data_in_memory=True, mat_file_path=None,
                 normalize_coordinates=False, remove_borders=True, extract_lines=False):
       
        self.keep_in_memory = keep_data_in_memory
        self.normalize_coords = normalize_coordinates
        self.remove_borders = remove_borders
        self.extract_lines = extract_lines

        self.vanishing_points_files = glob.glob(os.path.join(data_directory, "vps*"))
        self.lsd_line_files = glob.glob(os.path.join(data_directory, "lsd_lines*"))
        self.labelled_line_files = glob.glob(os.path.join(data_directory, "labelled_lines*"))
        self.vanishing_points_files.sort()
        self.lsd_line_files.sort()
        self.labelled_line_files.sort()

        if split == "train":
            self.set_ids = list(range(0, 1000))
        elif split == "val":
            self.set_ids = list(range(1000, 1224))
        elif split == "trainval":
            self.set_ids = list(range(0, 1224))
        elif split == "test":
            self.set_ids = list(range(1224, 1449))
        elif split == "all":
            self.set_ids = list(range(0, 1449))
        else:
            assert False, "invalid split: %s " % split

        self.dataset = [None for _ in self.set_ids]

        self.data_mat = None
        if mat_file_path is not None:
            self.data_mat = scipy.io.loadmat(mat_file_path, variable_names=["images"])

        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02

        K = np.matrix([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])

        if normalize_coordinates:
            S = np.matrix([[1. / 320., 0, -1.], [0, 1. / 320., -.75], [0, 0, 1]])
            K = S * K

        self.Kinv = K.I

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        
        id = self.set_ids[key]
        datum = self.dataset[key]

        if datum is None:

            lsd_line_segments = None

            if self.data_mat is not None:
                image_rgb = self.data_mat['images'][:, :, :, id]

                if self.remove_borders:
                    image_ = image_rgb[6:473, 7:631].copy()
                else:
                    image_ = image_rgb

                if self.extract_lines:
                    # Extract lines from the image using some method (not implemented here)
                    pass

            else:
                image_rgb = None

            if lsd_line_segments is None:
                lsd_line_segments = []

                if id < len(self.lsd_line_files):
                    with open(self.lsd_line_files[id], 'r') as csv_file:
                        reader = csv.DictReader(csv_file, delimiter=' ')
                        for line in reader:
                            p1x = float(line['point1_x'])
                            p1y = float(line['point1_y'])
                            p2x = float(line['point2_x'])
                            p2y = float(line['point2_y'])
                            lsd_line_segments += [np.array([p1x, p1y, p2x, p2y])]
                    lsd_line_segments = np.vstack(lsd_line_segments)
                else:
                    print("Index out of range: id =", id)

            labelled_line_segments = np.array([])
            with open(self.labelled_line_files[id], 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ')
                for line in reader:
                    lines_per_vp = np.array([])
                    for i in range(1, 5):
                        key_x1 = 'line%d_x1' % i
                        key_y1 = 'line%d_y1' % i
                        key_x2 = 'line%d_x2' % i
                        key_y2 = 'line%d_y2' % i

                        if line[key_x1] == '':
                            break

                        p1x = float(line[key_x1])
                        p1y = float(line[key_y1])
                        p2x = float(line[key_x2])
                        if line[key_y2] == '433q':
                            p2y = float(line[key_y2][0:-1])
                        else:
                            p2y = float(line[key_y2])

                        ls = np.array([p1x, p1y, p2x, p2y])
                        lines_per_vp = ls
                    lines_per_vp = np.vstack(lines_per_vp)
                    labelled_line_segments = lines_per_vp

            if self.normalize_coords:
                lsd_line_segments[:, 0] -= 320
                lsd_line_segments[:, 2] -= 320
                lsd_line_segments[:, 1] -= 240
                lsd_line_segments[:, 3] -= 240
                lsd_line_segments[:, 0:4] /= 320.

            line_segments = np.zeros((lsd_line_segments.shape[0], 7 + 2 + 3 + 3))
            for li in range(line_segments.shape[0]):
                p1 = np.array([lsd_line_segments[li, 0], lsd_line_segments[li, 1], 1])
                p2 = np.array([lsd_line_segments[li, 2], lsd_line_segments[li, 3], 1])
                centroid = 0.5 * (p1 + p2)
                line = np.cross(p1, p2)
                line /= np.linalg.norm(line[0:2])
                line_segments[li, 0:3] = p1
                line_segments[li, 3:6] = p2
                line_segments[li, 6:9] = line
                line_segments[li, 9:12] = centroid

            vp_pixel_list = []
            vp_homo_list = []
            with open(self.vanishing_points_files[id]) as csv_file:
                reader = csv.reader(csv_file, delimiter=' ')
                for ri, row in enumerate(reader):
                    if ri == 0:
                        continue
                    vp_pixel = np.array([float(row[1]), float(row[2])])
                    vp_pixel_list += [vp_pixel]

                    vp = vp_pixel.copy()
                    vp = np.concatenate((vp, np.ones((1,), dtype=np.float32)), axis=-1)
                    vp[0] -= 320
                    vp[1] -= 240
                    vp[1] *= -1
                    vp[0:2] /= 320.
                    vp /= np.linalg.norm(vp)
                    vp_homo_list += [vp]
            vps_pixel = np.vstack(vp_pixel_list)
            vps_homo = np.vstack(vp_homo_list)

            datum = {'line_segments': line_segments, 'vps_pixel': vps_pixel, 'id': id, 'vps_homo': vps_homo,
                     'image': image_rgb, 'labelled_lines': labelled_line_segments}

            if self.keep_in_memory:
                self.dataset[key] = datum

        return datum


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='NYU-VP dataset visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mat_file', default='C:/Users/Rabpreet/Desktop/nyu_depth_v2_labeled.v7.mat', help='mat_file')
    parser.add_argument('--data_dir', default="C:/Users/Rabpreet/Desktop/nyu_vp-master/nyu_vp-master/data", help='where to load')
    parser.add_argument('--save_dir', default='C:/Users/Rabpreet/Desktop/nyu_vp-master/nyu_vp-master/processed_data', help='where to save')
    opt = parser.parse_args()

    if opt.mat_file is None:
        print("Specify the path where your 'nyu_depth_v2_labeled.v7.mat' " +
              "is stored using the --mat_file option in order to load the original RGB images.")
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    dataset = NYUVanishingPointsDataset(data_directory=opt.data_dir, mat_file_path=opt.mat_file, split='all',
                                        normalize_coordinates=False, remove_borders=True, extract_lines=False)

    max_num_vp = 0
    for idx in range(len(dataset)):
        vpts_pixel = dataset[idx]['vps_pixel']
        vpts_homo = dataset[idx]['vps_homo']
        line_segments = dataset[idx]['line_segments']
        image = dataset[idx]['image']
        labelled_lines = dataset[idx]['labelled_lines']
        num_vps = vpts_pixel.shape[0]
        if num_vps > max_num_vp:
            max_num_vp = num_vps

        # Data augmentation
        for j in range(0, 4):
            if j == 0:
                image_name = os.path.join(opt.save_dir, f'{idx:04d}_0.png')
                sio.imsave(image_name, image)
                npz_name = os.path.join(opt.save_dir, f'{idx:04d}_0.npz')
                np.savez_compressed(npz_name, line_segments=line_segments, labelled_lines=labelled_lines,
                                    vpts_pixel=vpts_pixel, vpts=vpts_homo)

            if j == 1:  # left-right
                image_new = image.copy()[::, ::-1]
                labelled_lines_new = copy.deepcopy(labelled_lines)
                labelled_line = labelled_lines_new
                labelled_line[0] *= -1.
                labelled_line[0] += image.shape[1]
                labelled_line[2] *= -1.
                labelled_line[2] += image.shape[1]

                vpts_new = copy.deepcopy(vpts_homo)
                vpts_new[:, 0] *= -1.0

                image_name = os.path.join(opt.save_dir, f'{idx:04d}_1.png')
                sio.imsave(image_name, image_new)
                npz_name = os.path.join(opt.save_dir, f'{idx:04d}_1.npz')
                np.savez_compressed(npz_name, labelled_lines=labelled_lines_new, vpts=vpts_new)

            if j == 2:  # top-down
                image_new = image.copy()[::-1, ::]
                labelled_lines_new = copy.deepcopy(labelled_lines)
                labelled_line = labelled_lines_new
                labelled_line[1] *= -1.
                labelled_line[1] += image.shape[0]
                labelled_line[3] *= -1.
                labelled_line[3] += image.shape[0]

                vpts_new = copy.deepcopy(vpts_homo)
                vpts_new[:, 1] *= -1.0

                image_name = os.path.join(opt.save_dir, f'{idx:04d}_2.png')
                sio.imsave(image_name, image_new)
                npz_name = os.path.join(opt.save_dir, f'{idx:04d}_2.npz')
                np.savez_compressed(npz_name, labelled_lines=labelled_lines_new, vpts=vpts_new)

            if j == 3:  # top-down and left-right
                image_new = image.copy()[::-1, ::-1]
                labelled_lines_new = copy.deepcopy(labelled_lines)
                labelled_line = labelled_lines_new
                labelled_line[1] *= -1.
                labelled_line[1] += image.shape[0]
                labelled_line[3] *= -1.
                labelled_line[3] += image.shape[0]
                labelled_line[0] *= -1.
                labelled_line[0] += image.shape[1]
                labelled_line[2] *= -1.
                labelled_line[2] += image.shape[1]

                vpts_new = copy.deepcopy(vpts_homo)
                vpts_new[:, 1] *= -1.0
                vpts_new[:, 0] *= -1.0

                image_name = os.path.join(opt.save_dir, f'{idx:04d}_3.png')
                sio.imsave(image_name, image_new)
                npz_name = os.path.join(opt.save_dir, f'{idx:04d}_3.npz')
                np.savez_compressed(npz_name, labelled_lines=labelled_lines_new, vpts=vpts_new)
