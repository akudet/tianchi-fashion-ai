import collections
import time

import torch
import torch.utils.data as data
from torch.utils.data.dataset import random_split


def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, collections.Mapping):
        return {k: to_device(v, device) for k, v in item.items()}
    elif isinstance(item, collections.Sequence):
        return [to_device(d, device) for d in item]
    raise TypeError("not supported data type")


class Solver:
    def __init__(self, model, optim, loss_fn, metric_fn):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.optim = optim
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

    def fit(self, dataset, n_epochs, batch_size=32, train_split=1, print_cnt=0):
        """
        trains the model for a given number of epochs

        :param dataset:
            (Dataset) the dataset used for training, it item should be a pair of element, first is input, second is target
        :param n_epochs:
            (int), number of epochs to train the model
        :param batch_size:
            (int), number of samples per gradient update
        :param train_split:
            (int), between 0 and 1, fraction of data in dataset used for training
        :param print_cnt:
            (int), number of training info will be printed every epochs
        :return:
        """
        n_train = int(len(dataset) * train_split)
        n_val = len(dataset) - n_train
        dataset_train, dataset_val = random_split(dataset, [n_train, n_val])
        print("total:{}, n_train:{}, n_val:{}".format(len(dataset), len(dataset_train), len(dataset_val)))
        loader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        loader_val = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
        if print_cnt != 0:
            print_cnt = len(loader_train) // print_cnt + 1
        for i in range(1, n_epochs + 1):
            total_loss = 0
            cnt = 0
            n_cnt = print_cnt
            start_time = time.time()
            for x, y in loader_train:
                x = to_device(x, self.device)
                y = to_device(y, self.device)

                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                self.optim.step()
                cnt += 1
                if 0 < n_cnt <= cnt:
                    n_cnt += print_cnt
                    torch.save(self.model.state_dict(), "w.h5")
                    print("epoch:{} {:.1%}, loss:{}, time:{:.0f}s".format(i, cnt / len(loader_train), loss.item(),
                                                                          time.time() - start_time))

            # total_metric = 0
            # with torch.no_grad():
            #     for x, y in loader_val:
            #         x = x.to(self.device)
            #         y = y.to(self.device)
            #
            #         y_pred = self.model(x)
            #         total_metric += self.metric_fn(y_pred, y)
            #
            # print("epoch:{} loss:{} metric:{}".format(i, total_loss / n_train, total_metric / n_val))
