from os import path, listdir
import torch
from torchvision import transforms
import random

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


colors_per_class = {
    'picture_1' : [254, 202, 87],
    'picture_2' : [255, 107, 107],
    'picture_3' : [10, 189, 227],
    'picture_4' : [255, 159, 243],
    'picture_5' : [16, 172, 132],
    'picture_6' : [128, 80, 128],
    'picture_7' : [87, 101, 116],
    'picture_8' : [52, 31, 151],
    'picture_9' : [0, 0, 0],
    'picture_10' : [100, 100, 255],
    'picture_11' : [34, 31, 151],
    'picture_12' : [56, 89, 255],
    'picture_13' : [120, 100, 255],
    'picture_14' : [10, 67, 255],

}


# processes Animals10 dataset: https://www.kaggle.com/alessiocorrado99/animals10
class PetroleumDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, num_images=1000):
        translation = {'fourteen' : 'picture_1',
                       'one' : 'picture_2',
                       'two' : 'picture_3',
                       'three' : 'picture_4',
                       'four': 'picture_5',
                       'five': 'picture_6',
                       'six': 'picture_7',
                       'seven': 'picture_8',
                       'eight': 'picture_9',
                       'nine': 'picture_10',
                       'ten': 'picture_11',
                       'eleven': 'picture_12',
                       'twelve': 'picture_13',
                       'therteen': 'picture_14',

                       }

        self.classes = translation.values()
        print(data_path)
        if not path.exists(data_path):
            raise Exception(data_path + ' does not exist!')

        self.data = []

        folders = listdir(data_path)
        for folder in folders:
            # print(folder)
            label = translation[folder]
            # print(data_path)
            full_path = path.join(data_path, folder)
            images = listdir(full_path)
            # print(images)

            current_data = [(path.join(full_path, image), label) for image in images]
            # print(current_data)
            self.data += current_data
            # print(len(self.data))

        num_images = min(num_images, len(self.data))
        self.data = random.sample(self.data, num_images) # only use num_images images

        # We use the transforms described in official PyTorch ResNet inference example:
        # https://pytorch.org/hub/pytorch_vision_resnet/.
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        image_path, label = self.data[index]

        image = Image.open(image_path)
        # print('======+++++++========')
        # print(image)
        # print('____======_____')

        try:
            image = self.transform(image) # some images in the dataset cannot be processed - we'll skip them
            # print(image)
        except Exception:
            return None

        dict_data = {
            'image' : image,
            'label' : label,
            'image_path' : image_path
        }
        # print('+++++++++++++++++++++++++++++=')
        # print(dict_data)
        return dict_data


# Skips empty samples in a batch
def collate_skip_empty(batch):
    batch = [sample for sample in batch if sample] # check that sample is not None
    return torch.utils.data.dataloader.default_collate(batch)