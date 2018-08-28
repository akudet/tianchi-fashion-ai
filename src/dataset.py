import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class ApplyTo:
    """
    apply a specific function to an element of a map
    """

    def __init__(self, key, func):
        self.func = func
        self.key = key

    def __call__(self, m):
        m[self.key] = self.func(m[self.key])
        return m


class KeyPoints:
    """
    take fashion ai keypoints and generate a array of keypoints in the order given by kpt_names,
    only keypoints in kpt_names are considered
    also make x,y start from 0 instead of 1
    """

    def __init__(self, kpt_names):
        self.kpt_names = kpt_names

    def __call__(self, item):
        kpts = np.zeros((len(self.kpt_names), 3))
        for i, attr_name in enumerate(self.kpt_names):
            kpts[i] = item[attr_name]
        # attribute keypoint x,y is start from 1
        kpts[:, :2] -= 1
        item["kpts"] = kpts
        return item


class Mask:
    """
    take an image category and generate a list of mask for each kpt in the order given by kpt_names
    only keypoints in kpt_names are considered
    """

    # keypoints of each category
    kpt_names_by_cat = {
        'blouse': {
            'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
            'armpit_left', 'armpit_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
            'cuff_right_out', 'top_hem_left', 'top_hem_right'
        },
        'outwear': {
            'neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'armpit_left',
            'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out',
            'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right'
        },
        'trousers': {
            'waistband_left', 'waistband_right', 'crotch', 'bottom_left_in', 'bottom_left_out',
            'bottom_right_in', 'bottom_right_out'
        },
        'skirt': {
            'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right'
        },
        'dress': {
            'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left',
            'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
            'cuff_right_out', 'hemline_left', 'hemline_right'
        }}

    def __init__(self, kpt_names):
        self.kpt_names = kpt_names

    def __call__(self, item):
        # this will do a proper broadcasting for heatmap(C,H,W)
        mask = torch.zeros((len(self.kpt_names), 1, 1))
        kpt_names = self.kpt_names_by_cat[item["image_category"]]
        for i, kpt_name in enumerate(self.kpt_names):
            if kpt_name in kpt_names:
                mask[i] = 1

        item["mask"] = mask
        return item


class AnnotatedMask:
    """take a list of keyoints generate a list of mask of annotated keypoints in the order given by kpt_names"""

    def __init__(self, kpt_names):
        self.kpt_names = kpt_names

    def __call__(self, item):
        mask = torch.zeros((len(self.kpt_names), 1, 1))
        for i, kpt_name in enumerate(self.kpt_names):
            kpt = item[kpt_name]
            if kpt[2] != -1:
                mask[i] = 1

        item["mask"] = mask
        return item


class Resize(transforms.Resize):

    def __call__(self, item):
        ow, oh = item["image"].size
        item["image"] = super().__call__(item["image"])
        w, h = item["image"].size
        item["kpts"][:, 0] *= w / ow
        item["kpts"][:, 1] *= h / oh
        return item


class Heatmap:
    """
    take list of fashion ai keypoints to generate a heatmap
    """

    def __call__(self, item):
        scale = 4
        w, h = item["image"].size
        kpts = item["kpts"].astype(np.int32)
        heatmap = torch.zeros(len(kpts), h // scale, w // scale)
        for i, attr in enumerate(kpts):
            if attr[2] != -1:
                x, y, visibility = attr
                heatmap[i, y // scale, x // scale] = 1
        item["heatmap"] = heatmap
        return item


class HeatmapToKeyPoints:

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, heatmap, mask, size):
        N, C, H, W = heatmap.shape
        kpts = heatmap.new_zeros((N, C, 3), torch.float32)
        heatmap = heatmap.view(N, C, -1)
        confidence, position = torch.max(heatmap, dim=2)
        confidence = confidence.view(N, C)
        kpts[:, :, 2] = confidence > self.threshold
        kpts[:, :, 0] = (position % W)
        kpts[:, :, 1] = (position // H)
        size = size.to(torch.float32)
        size = size.view(N, 1, 2)
        size[:, :, 0] /= W
        size[:, :, 1] /= H
        kpts[:, :, :2] *= size
        kpts[:, :, :2] += size / 2
        # target x,y start from 1, in training x,y is start from 0, see KeyPoints class
        kpts[:, :, :2] += 1

        mask = mask.view(N, C, 1)
        kpts *= mask
        kpts += mask - 1
        kpts = kpts.to(torch.int32)
        return kpts


def fashion_ai_dataset(root, anno_csv, is_train=True):
    """
    factor method used to get fashion ai dataset for training and testing

    :param root:
        the root of image folder, which the image_id(relative_path) refer to
    :param anno_csv:
        the path of annotation csv file
    :param is_train:
        if True, item will be (image, (heatmap, mask, origin_img_size)), otherwise (image, (mask, origin_img_size))
    :return:
    """

    dataset = FashionAIDataset(root, anno_csv)
    kpt_names = dataset.kpt_names
    if is_train:
        transform = transforms.Compose([
            KeyPoints(kpt_names),
            Resize((256, 256)),
            Heatmap(),
            ApplyTo("image", transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])),
            AnnotatedMask(kpt_names),
            lambda item: (item["image"], (item["heatmap"], item["mask"], item["size"])),
        ])
    else:
        transform = transforms.Compose([
            ApplyTo("image", transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])),
            Mask(kpt_names),
            lambda item: (item["image"], (item["mask"], item["size"])),
        ])

    dataset = TransformDataset(dataset, transform)


    return dataset


class FashionAIDataset(data.Dataset):
    kpt_names = [
        "neckline_left", "neckline_right",
        "center_front",
        "shoulder_left", "shoulder_right",
        "armpit_left", "armpit_right",
        "waistline_left", "waistline_right",
        "cuff_left_in", "cuff_left_out", "cuff_right_in", "cuff_right_out",
        "top_hem_left", "top_hem_right",
        "waistband_left", "waistband_right",
        "hemline_left", "hemline_right",
        "crotch",
        "bottom_left_in", "bottom_left_out", "bottom_right_in", "bottom_right_out",
    ]

    def __init__(self, root, anno_csv):
        from torchvision.datasets.folder import default_loader

        df = pd.read_csv(anno_csv)

        for fashion_attr in df.columns[2:]:
            df[fashion_attr] = df[fashion_attr].apply(lambda attr: list(map(int, attr.split("_"))))

        df["image_path"] = [os.path.join(root, image_id) for image_id in df["image_id"]]

        self.df = df
        self.loader = default_loader

    def __getitem__(self, index):
        item = self.df.iloc[int(index)].to_dict()
        item["image"] = self.loader(item["image_path"])
        item["size"] = torch.Tensor(item["image"].size)
        return item

    def __len__(self):
        return len(self.df)


class TransformDataset(data.Dataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.dataset[index])

    def __len__(self):
        return len(self.dataset)
