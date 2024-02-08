import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import random
import numpy as np
import math


def seg_norm_lpba40(seg):
    seg_table = np.array([0,21,22,23,24,25,26,27,28,29,30,31,32,33,34,41,42,43,
                          44,45,46,47,48,49,50,61,62,63,64,65,66,67,68,81,82,83,
                          84,85,86,87,88,89,90,91,92,101,102,121,122,161,162,163,
                          164,165,166,181,182])
    seg_out = np.zeros_like(seg)
    for i in range(len(seg_table)):
        seg_out[seg == seg_table[i]] = i
    return seg_out


class LPBA40Dataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        img = nib.load(path)
        label = nib.load(path.replace("Train", "Label"))
        x = img.get_fdata()
        x_seg = seg_norm_lpba40(label.get_fdata())

        img_ = nib.load(tar_file)
        label_ = nib.load(tar_file.replace("Train", "Label"))
        y = img_.get_fdata()
        y_seg = seg_norm_lpba40(label_.get_fdata())

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


class LPBA40_InferDataset(Dataset):
    def __init__(self, data_path, transforms, istest=False):
        self.paths = data_path
        self.transforms = transforms
        self.istest = istest

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x_index = index // (len(self.paths) - 1)
        s = index % (len(self.paths) - 1)
        y_index = s + 1 if s >= x_index else s
        path = self.paths[x_index]
        tar_file = self.paths[y_index]

        a_name = int(os.path.split(path)[1].split(".")[0].split("S")[-1])
        b_name = int(os.path.split(tar_file)[1].split(".")[0].split("S")[-1])
        infer_save_name = "LPBA_" + str(a_name) + "to" + str(b_name) + ".nii.gz"

        img = nib.load(path)
        if self.istest:
            label = nib.load(path.replace("Test", "Label"))
            label_ = nib.load(tar_file.replace("Test", "Label"))
        else:
            label = nib.load(path.replace("Val", "Label"))
            label_ = nib.load(tar_file.replace("Vrain", "Label"))
        x = img.get_fdata()
        x_seg = seg_norm_lpba40(label.get_fdata())

        img_ = nib.load(tar_file)
        y = img_.get_fdata()
        y_seg = seg_norm_lpba40(label_.get_fdata())

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg, infer_save_name

    def __len__(self):
        return len(self.paths) * (len(self.paths) - 1)


