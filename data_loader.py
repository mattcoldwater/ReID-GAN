import torch
import torchvision.datasets as dsets
from torchvision import transforms

from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils import RandomErasing, RandomSampler
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os
import re
import random
import json
from pathlib import Path

class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes=['church_outdoor_train','classroom_train']):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=classes, transform=transforms)
        return dataset
    
    def load_imagenet(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset

    def load_off(self):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'imagenet':
            dataset = self.load_imagenet()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'off':
            dataset = self.load_off()

        print('dataset',len(dataset))
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=True)
        return loader

def stratify_sample(labels, seed, val_size):
    X = np.arange(len(labels))
    y = np.asarray(labels)
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=val_size,
                                                    random_state=seed)
    return X_train, X_val

class Data():
    def __init__(self, opt):
        train_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if opt.dataset == "viva":
            pre_viva = Vivalab_pre()
            # pre_viva.check_tids(opt.image_path)
            train_imgs, val_imgs, test_imgs, query_imgs, self.tid_dict, self.test_query_imgs = pre_viva.get_all(opt)
            self.test_transform = test_transform
            self.trainset = Vivalab(train_transform, train_imgs, 'train', opt.trans)
            self.valset = Vivalab(train_transform, val_imgs, 'val', opt.trans)
            self.testset = Vivalab(test_transform, test_imgs, 'test', opt.trans)
            self.queryset = Vivalab(test_transform, query_imgs, 'query', opt.trans)
        else:
            self.trainset = Market1501(train_transform, 'train', opt.image_path, opt.seed)
            self.valset = Market1501(train_transform, 'val', opt.image_path, opt.seed)
            self.testset = Market1501(test_transform, 'test', opt.image_path, opt.seed)
            self.queryset = Market1501(test_transform, 'query', opt.image_path, opt.seed)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid, batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=opt.num_workers,
                                                  pin_memory = True)
        self.val_loader = dataloader.DataLoader(self.valset, batch_size=opt.batchtest, num_workers=opt.num_workers, pin_memory = True, drop_last=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=opt.num_workers, pin_memory = True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=opt.num_workers, pin_memory = True)

class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path, seed):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype in ['train', 'val']:
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        img_ids = [self.id(path) for path in self.imgs]
        
        if dtype in ['train', 'val']:
            train_index, val_index = stratify_sample(img_ids, seed, 0.1)
            if dtype == "train":
                self.imgs = [self.imgs[i] for i in train_index]
            elif dtype == "val":
                self.imgs = [self.imgs[i] for i in val_index]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])

class Vivalab(dataset.Dataset):
    def __init__(self, transform, imgs, dtype='', trans=1):
        self.transform = transform
        if trans == 0:
            self.loader = self.img_loader
        else:
            self.loader = default_loader
        
        self.imgs = imgs               
        
        if dtype != '': print(dtype, len(self.imgs))

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

        self.cameras = [self.camera(path) for path in self.imgs]

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def img_loader(file_path):
        with open(file_path, 'rb') as f:
            _ = Image.open(f)
            _.load()
            image = Image.new("RGB", _.size, "WHITE")
            image.paste(_, (0, 0), _)
        _ = None
        del _
        return image

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        p = file_path.with_suffix('.json')
        with p.open() as f:
            temp = json.loads(f.read())
            c = int(temp['CAMERA'])
        return c

    @staticmethod
    def tid(file_path):
        """
        :param file_path: unix style file path
        :return: tracklet id
        """
        return int(file_path.parts[-2])

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """

        return int(file_path.parts[-3])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    def list_pictures(self, path):
        paths = Path(path).glob('**/*.png')
        return paths

class Vivalab_pre():
    def __init__(self):
        pass

    @staticmethod
    def list_pictures(path):
        paths = Path(path).glob('**/*.png')
        return paths   

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.parts[-3])

    @staticmethod
    def tid(file_path):
        """
        :param file_path: unix style file path
        :return: tracklet id
        """
        return int(file_path.parts[-2])

    @staticmethod
    def check_tids(datadir):
        data_path = datadir
        all_ids = os.listdir(data_path)     
        all_tids = []
        for _ in all_ids:
            all_tids += os.listdir(data_path+'/'+_)
        assert len(all_tids) == len(set(all_tids))

    def get_all(self, opt):
        data_path = opt.image_path
        all_ids = os.listdir(data_path)
        all_ids = [int(_) for _ in all_ids]
        imgs = [path for path in self.list_pictures(data_path)]
        
        VRandom = random.Random(opt.seed)
        VRandom.shuffle(all_ids)
        VRandom.shuffle(imgs)

        assert all_ids[:14] == [34073, 34530, 34368, 34088, 33966, 34382, 34475, 33897, 34362, 34061, 34437, 34254, 34166, 34089]

        _len = len(all_ids)
        assert _len // 2 == opt.num_classes
        print("num of categories", _len)

        ## train & val
        train_val_ids = all_ids[:_len//2]
        train_val_imgs = [path for path in imgs if self.id(path) != -1 and self.id(path) in train_val_ids]
        labels = [self.id(path) for path in train_val_imgs]
        train_index, val_index = stratify_sample(labels, opt.seed, 0.1)
        train_imgs = [train_val_imgs[i] for i in train_index]  # 98797
        val_imgs = [train_val_imgs[i] for i in val_index] # 10978

        ## test query
        test_query_ids = all_ids[_len//2:]
        if opt.debug: test_query_ids = test_query_ids[:10]
        test_query_imgs = [path for path in imgs if self.id(path) != -1 and self.id(path) in test_query_ids]

        ## id dict
        id_dict = {}
        for path in test_query_imgs:
            path_id = self.id(path) 
            if path_id in id_dict:
                if len(id_dict[path_id]) < 80:
                    id_dict[path_id].append(path)
            else:
                id_dict[path_id] = [path,]
     
        ## test & query
        test_imgs, query_imgs  = [], []
        # tmp = [len(y) for y in list(id_dict.values())]
        # print(max(tmp), min(tmp)) # 20 5733
        for path_id in list(id_dict.keys()):
            paths = id_dict[path_id]
            # VRandom.shuffle(paths) have been shuffled before for the whole dataset
            _n = round(len(paths) * 0.1419)
            query_imgs += paths[:_n] # 2166
            test_imgs += paths[_n:]  # 13555
        

        # test_query_imgs = test_query_imgs[:10000]
        

        ## tid_dict
        test_query_tids = np.array([self.tid(path) for path in test_query_imgs])
        arg_ = np.argsort(test_query_tids)
        test_query_imgs = np.array(test_query_imgs)
        test_query_imgs = test_query_imgs[arg_]
        test_query_imgs = test_query_imgs.tolist()
        del arg_, test_query_tids
        tid_dict = {}
        for i in range(len(test_query_imgs)):
            path = test_query_imgs[i]
            path_tid = self.tid(path) 
            if path_tid not in tid_dict:
                tid_dict[path_tid] = i

        return train_imgs, val_imgs, test_imgs, query_imgs, tid_dict, test_query_imgs