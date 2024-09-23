import torch
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
import glob
import os
import SimpleITK as sitk
import random
import numpy as np
# from . import image_transforms as myit
from .specifics import *


class TestDataset(Dataset):

    def __init__(self, args, files, transforms=None):

        self.transforms = transforms
        self.files = files
        
        # reading the paths
        # self.files = glob.glob(os.path.join(args.data_root, 'images/image*'))
        # self.files = sorted(self.files, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        # self.files = self.files[:len(self.files)//60]

        # self.FOLD = get_folds(args.dataset)
        # self.files = [elem for idx, elem in enumerate(self.files) if idx in self.FOLD[args.fold]]

        # split into support/query\
        self.support_index = random.randint(0,len(self.files)-1)
        self.support_dir = self.files[self.support_index] # - 1
        self.files = [self.files[image] for image in range(len(self.files)) if image != self.support_index] # :-1  # remove support 
        self.label_id = None 


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # img_path = self.files[idx]
        # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        # img = (img - img.mean()) / img.std()

        di = self.transforms(self.files[idx])
        img = di['image'][0].cpu().numpy() # SELECT flair
        label = di['label'][0].cpu().numpy()

        # transpose image and label, which are (H, W, D) to (D, H, W)
        img = np.transpose(img, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        img = np.stack(3 * [img], axis=1)

        # lbl = sitk.GetArrayFromImage(
        #     sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))
        # lbl[lbl == 200] = 1
        # lbl[lbl == 500] = 2
        # lbl[lbl == 600] = 3

        # label = 1 * (label == self.label_id) # ?
        # print("files", self.files[idx])
        id_value = self.files[idx]["label"].split('/')[-1].split('-')[2]
        print("id value", id_value)
        sample = {'id': id_value}

        sample['image'] = img
        sample['label'] = label

        return sample

    def get_support_index(self, n_shot, C):
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label class!')

        
        di = self.transforms(self.support_dir)
        img = di['image'][0] # select FLAIR only
        label = di['label'][0]

        # transpose image and label, which are (H, W, D) to (D, H, W)
        img = np.transpose(img, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        # stack three times on axis 1
        img = torch.stack(3 * [img], axis=1)

        print("image shape", img.shape)

        sample = {}
        if all_slices:
            sample['image'] = img
            sample['label'] = label
        else:
            # select N labeled slices
            if N is None:
                raise ValueError('Need to specify number of labeled slices!')
            idx = label.sum(axis=(1, 2)) > 0
            idx_ = self.get_support_index(N, idx.sum())

            sample['image'] = img[idx][idx_]
            sample['label'] = label[idx][idx_]

        return sample


class TrainDataset(Dataset):

    def __init__(self, args, files, transforms=None):
        self.n_shot = args.n_shot
        self.n_way = args.n_way
        self.n_query = args.n_query
        self.n_sv = args.n_sv
        # self.max_iter = args.max_iterations
        self.read = True  # read images before get_item
        self.train_sampling = 'neighbors'
        self.min_size = 200
        self.factor = 20

        self.files = files
        self.transforms = transforms

        # self.files = glob.glob(os.path.join(args.data_root, 'images/image*'))
        # self.files = sorted(self.files, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        # self.files = self.files[:len(self.files)//self.factor]

        # self.label_dirs = glob.glob(os.path.join(args.data_root, 'labels/label*'))
        # self.label_dirs = sorted(self.label_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        # self.label_dirs = self.label_dirs[:len(self.label_dirs)//self.factor]

        # remove test fold!
        # self.FOLD = get_folds(args.dataset)
        # self.files = [elem for idx, elem in enumerate(self.files) if idx not in self.FOLD[args.fold]]
        # self.label_dirs = [elem for idx, elem in enumerate(self.label_dirs) if idx not in self.FOLD[args.fold]]

        # read images
        # if self.read:
        #     self.images = {}
        #     self.labels = {}
        #     for image_dir, label_dir in zip(self.files, self.label_dirs):
        #         self.images[image_dir] = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
        #         self.labels[label_dir] = sitk.GetArrayFromImage(sitk.ReadImage(label_dir))

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):

        # sample patient idx
        # pat_idx = random.choice(range(len(self.files)))

        # if self.read:
        #     # get image/label volume from dictionary
        #     img = self.images[self.files[pat_idx]]
        #     label = self.labels[self.label_dirs[pat_idx]]
        # else:
        #     # read image/supervoxel volume into memory
        #     img = sitk.GetArrayFromImage(sitk.ReadImage(self.files[pat_idx]))
        #     label = sitk.GetArrayFromImage(sitk.ReadImage(self.label_dirs[pat_idx]))

        # # normalize
        # img = (img - img.mean()) / img.std()

        # sample class(es) (supervoxel)

        di = self.transforms(self.files[idx])

        img = di['image'].cpu().numpy()[0] # select flair
        label = di['label'].cpu().numpy()[0]

        # transpose image and label, which are (H, W, D) to (D, H, W)
        img = np.transpose(img, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        # print("image, label shape", img.shape, label.shape)

        # only select t2 for the image, which happens to be the first index for all images

        # img = np.transpose(img, (0, 3, 1, 2))
        # label = np.transpose(label, (0, 3, 1, 2)) # B x D x H x W
        # print("image, label shape", img.shape, label.shape)

        # print label uniques
        # print(np.unique(label))

        # transpose
        

        
        unique = list(np.unique(label))
        unique.remove(0)

        size = 0
        while size < self.min_size:
            n_slices = (self.n_shot * self.n_way) + self.n_query - 1
            while n_slices < ((self.n_shot * self.n_way) + self.n_query):
                cls_idx = random.choice(unique)

                # extract slices containing the sampled class
                sli_idx = np.sum(label == cls_idx, axis=(1, 2)) > 0
                n_slices = np.sum(sli_idx)

            img_slices = img[sli_idx]
            label_slices = 1 * (label[sli_idx] == cls_idx)

            # sample support and query slices
            i = random.choice(
                np.arange(n_slices - ((self.n_shot * self.n_way) + self.n_query) + 1))  # successive slices
            sample = np.arange(i, i + (self.n_shot * self.n_way) + self.n_query)

            size = np.sum(label_slices[sample[0]])

        # invert order
        if np.random.random(1) > 0.5:
            sample = sample[::-1]  # successive slices (inverted)

        sup_lbl = label_slices[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        qry_lbl = label_slices[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W

        sup_img = img_slices[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        sup_img = np.stack((sup_img, sup_img, sup_img), axis=2)
        qry_img = img_slices[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W
        qry_img = np.stack((qry_img, qry_img, qry_img), axis=1)

        # gamma transform
        # if np.random.random(1) > 0.5:
        #     qry_img = self.gamma_tansform(qry_img)
        # else:
        #     sup_img = self.gamma_tansform(sup_img)

        # # geom transform
        # if np.random.random(1) > 0.5:
        #     qry_img, qry_lbl = self.geom_transform(qry_img, qry_lbl)
        # else:
        #     sup_img, sup_lbl = self.geom_transform(sup_img, sup_lbl)

        # print shapes of support, query, labels
        # print("support image shape", sup_img.shape)
        # print("query image shape", qry_img.shape)
        # print("support label shape", sup_lbl.shape)
        # print("query label shape", qry_lbl.shape)

        sample = {'support_images': sup_img,
                  'support_fg_labels': sup_lbl,
                  'query_images': qry_img,
                  'query_labels': qry_lbl,
                  'id': self.files[idx]["label"].split('/')[-1].split('-')[2]}

        return sample

