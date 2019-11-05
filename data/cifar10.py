# -*- coding: utf-8 -*-

from data.transform import img_transform
from data.transform import Onehot

import numpy as np
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataloader import DataLoader

import os
import sys
import pickle
import gensim

def load_data(opt):
    """加载cifar10数据
    Parameters
        opt: Parser
        配置
    Returns
        query_dataloader, train_dataloader, database_dataloader: DataLoader
        数据加载器
    """
    CIFAR10.init(opt.data_path, opt.semantic_path, opt.unseen_class, opt.num_query, opt.num_train)
    query_dataset = CIFAR10('query', transform=img_transform(), target_transform=Onehot())
    train_dataset = CIFAR10('train', transform=img_transform(), target_transform=Onehot())
    database_dataset = CIFAR10('database', transform=img_transform(), target_transform=Onehot())

    query_dataloader = DataLoader(query_dataset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  )
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  )
    database_dataloader = DataLoader(database_dataset,
                                     batch_size=opt.batch_size,
                                     num_workers=opt.num_workers,
                                     )

    return query_dataloader, train_dataloader, database_dataloader

class CIFAR10(data.Dataset):
    """加载官网下载的CIFAR10数据集"""
    @staticmethod
    def init(root_dir, semantic_path,unseen_class, num_query, num_train):
        name2label = {'airplane': 0,
                      'automobile': 1,
                      'bird': 2,
                      'cat': 3,
                      'deer': 4,
                      'dog': 5,
                      'frog': 6,
                      'horse': 7,
                      'ship': 8,
                      'truck': 9,
                      }
        label2name = {v:k for k,v in name2label.items()}
        data_list = ['data_batch_1',
                     'data_batch_2',
                     'data_batch_3',
                     'data_batch_4',
                     'data_batch_5',
                     'test_batch',
                     ]
        base_folder = 'cifar-10-batches-py'

        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(semantic_path,binary=True)

        data = []
        targets = []
        semantics = []

        for file_name in data_list:
            file_path = os.path.join(root_dir, base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                    for label in entry['labels']:
                        semantics.append(w2v_model[label2name[label] ])
                else:
                    targets.extend(entry['fine_labels'])
                    for label in entry['fine_labels']:
                        semantics.append(w2v_model[label2name[label] ])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.array(targets)
        semantics = np.array(semantics)

        CIFAR10.ALL_IMG = data
        CIFAR10.ALL_TARGETS = targets
        CIFAR10.ALL_SEMANTICS = semantics

        # sort by class
        sort_index = CIFAR10.ALL_TARGETS.argsort()
        CIFAR10.ALL_IMG = CIFAR10.ALL_IMG[sort_index, :]
        CIFAR10.ALL_TARGETS = CIFAR10.ALL_TARGETS[sort_index]
        CIFAR10.ALL_SEMANTICS = CIFAR10.ALL_SEMANTICS[sort_index]

        # sample query from unseen class
        per_class_num = CIFAR10.ALL_IMG.shape[0] // 10
        unseen_label = name2label[unseen_class]
        unseen_index = np.arange(unseen_label*per_class_num, (unseen_label+1)*per_class_num)
        query_index = np.random.permutation(unseen_index)[:num_query]
        # sample train from seen class
        seen_index = np.array(list(set(range(CIFAR10.ALL_IMG.shape[0])).difference(set(unseen_index))))
        train_index = np.random.permutation(seen_index)[:num_train]
        # remain data (except query) for retrivel database 
        db_index = np.array(list(set(range(CIFAR10.ALL_IMG.shape[0])).difference(set(query_index))))
        db_index = np.random.permutation(db_index)

        # split data, tags
        CIFAR10.QUERY_IMG = CIFAR10.ALL_IMG[query_index, :]
        CIFAR10.QUERY_TARGETS = CIFAR10.ALL_TARGETS[query_index]
        CIFAR10.QUERY_SEMANTICS = CIFAR10.ALL_SEMANTICS[query_index]
        CIFAR10.TRAIN_IMG = CIFAR10.ALL_IMG[train_index, :]
        CIFAR10.TRAIN_TARGETS = CIFAR10.ALL_TARGETS[train_index]
        CIFAR10.TRAIN_SEMANTICS = CIFAR10.ALL_SEMANTICS[train_index]
        CIFAR10.DB_IMG = CIFAR10.ALL_IMG[db_index, :]
        CIFAR10.DB_TARGETS = CIFAR10.ALL_TARGETS[db_index]
        CIFAR10.DB_SEMANTICS = CIFAR10.ALL_SEMANTICS[db_index]

    def __init__(self, mode='train',
                 transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.img = CIFAR10.TRAIN_IMG
            self.targets = CIFAR10.TRAIN_TARGETS
            self.semantics = CIFAR10.TRAIN_SEMANTICS
        elif mode == 'query':
            self.img = CIFAR10.QUERY_IMG
            self.targets = CIFAR10.QUERY_TARGETS
            self.semantics = CIFAR10.QUERY_SEMANTICS
        elif mode == 'database':
            self.img = CIFAR10.DB_IMG
            self.targets = CIFAR10.DB_TARGETS
            self.semantics = CIFAR10.DB_SEMANTICS

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target, semantics = self.img[index], self.targets[index], self.semantics[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semantics, index

    def __len__(self):
        return len(self.img)
