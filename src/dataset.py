import os

import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class ApplyTo:

    def __init__(self, key, func):
        self.func = func
        self.key = key

    def __call__(self, m):
        m[self.key] = self.func(m[self.key])
        return m


class AttributesHeatmap:
    """
    take list of fashion ai attributes to generate a heatmap
    """

    def __init__(self, attr_names):
        self.attr_names = attr_names

    def __call__(self, item):
        w, h = item["image"].size
        heatmap = torch.zeros((len(self.attr_names), h, w))
        for i, attr_name in enumerate(self.attr_names):
            attr = item[attr_name]
            if attr[2] != -1:
                x, y, visibility = attr
                heatmap[i, y, x] = 1
                # heatmap[i, y, x] = visibility

        item["heatmap"] = heatmap
        return item


def fashion_ai_dataset(root, anno_csv):
    dataset = FashionAIDataset(root, anno_csv)

    transform = transforms.Compose([
        AttributesHeatmap(dataset.df.columns[2:2 + 24]),
        ApplyTo("image", transforms.ToTensor()),
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
        item = self.df.iloc[index].to_dict()
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
