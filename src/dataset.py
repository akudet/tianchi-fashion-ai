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


class Attributes:
    """
    convert fashion ai attributes keypoints to a numpy array, also make x,y start from 0 instead of 1
    """

    def __init__(self, attr_names):
        self.attr_names = attr_names

    def __call__(self, item):
        attrs = np.zeros((len(self.attr_names), 3))
        for i, attr_name in enumerate(self.attr_names):
            attrs[i] = item[attr_name]
        # attribute keypoint x,y is start from 1
        attrs[:, :2] -= 1
        item["attrs"] = attrs
        return item


class Resize(transforms.Resize):

    def __call__(self, item):
        ow, oh = item["image"].size
        item["image"] = super().__call__(item["image"])
        w, h = item["image"].size
        item["attrs"][:, 0] *= w / ow
        item["attrs"][:, 1] *= h / oh
        return item


class Heatmap:
    """
    take list of fashion ai attributes to generate a heatmap
    """

    def __call__(self, item):
        scale = 4
        w, h = item["image"].size
        attrs = item["attrs"].astype(np.int32)
        heatmap = torch.zeros(len(attrs), h // scale, w // scale)
        for i, attr in enumerate(attrs):
            if attr[2] != -1:
                x, y, visibility = attr
                heatmap[i, y // scale, x // scale] = 1
                # heatmap[i, y, x] = visibility
        item["heatmap"] = heatmap
        return item


def fashion_ai_dataset(root, anno_csv):
    """
    factor method used to get fashion ai dataset for training and testing

    :param root:
        the root of image folder, which the image_id(relative_path) refer to
    :param anno_csv:
        the path of annotation csv file
    :return:
    """

    dataset = FashionAIDataset(root, anno_csv)

    attr_names = dataset.df.columns[2:2 + 24]

    transform = transforms.Compose([
        Attributes(attr_names),
        Resize((256, 256)),
        Heatmap(),
        ApplyTo("image", transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])),
        lambda item: (item["image"], item["heatmap"]),
    ])

    dataset = TransformDataset(dataset, transform)

    return dataset


class FashionAIDataset(data.Dataset):

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
