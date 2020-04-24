import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

class ImageTransform():
    """
    画像の前処理クラス。訓練時、検証時で異なる動作
    画像をリサイズし色を標準化
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする

    Attriburtes
    ------------
    resize : int
        リサイズ先の画像の大きさ
    mean : (R, G, B)
        各色チャンネルの平均値
    std : (R, G, B)
        各色チャンネルの標準偏差
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train" : transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "val" : transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase="train"):
        """
        Parameters
        ----------
        phase : "train" or "val"
            前処理のモードを指定
        """
        return self.data_transform[phase](img)


def make_datapath_list(phase="train"):
    """
    データパスを格納したリストを作成する

    Parameters
    ----------
    phase : "train" or "val"
        訓練データか検証データかを指定する

    Returns
    -------
    path_list : list
        データへのパスを格納したリスト
    """

    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath + phase + "/**/*.jpg")
    print(target_path)

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class HymenopteraDataset(data.Dataset):
    """
    アリとハチの画像のDatasetクラス。PyTorchのDatasetクラスを継承

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : "train" or "val"
        学習か訓練か指定
    """

    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        """画像の枚数を返す"""
        return len(self.file_list)

    def __getitem__(self, index):
        """
        前処理した画像のTensor形式のデータとラベルを取得
        """

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path) #[高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(
            img, self.phase) # torch.Size([3, 224, 224])

        #画像のラベルをファイル名から抜き出す
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]

        # ラベルを数値に変更する
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label
