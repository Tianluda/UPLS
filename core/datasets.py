import os
import torch
from PIL import Image
from tools.ai.augment_utils import *
import torchvision.transforms as transforms

import imageio
import numpy as np
from PIL import Image
from tools.ai.augment_utils import *
from tools.ai.torch_utils import one_hot_embedding
from tools.general.xml_utils import read_xml
from tools.general.json_utils import read_json
from tools.dataset.voc_utils import get_color_map_dic,color_map

class CUSTOM_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir1, data_dir2=None,data_dir3=None, transform=None):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.data_dir3 = data_dir3
        self.transform = transform
        self.image_name_list = []
        # 读取第一个目录中的图像
        self.image_name_list += [os.path.join(self.data_dir1, f) for f in os.listdir(self.data_dir1) if f.endswith('.jpg') or f.endswith('.png')]
        # 如果有第二个目录，读取其中的图像
        if self.data_dir2:
            self.image_name_list += [os.path.join(self.data_dir2, f) for f in os.listdir(self.data_dir2) if f.endswith('.jpg') or f.endswith('.png')]
        if self.data_dir3:
            self.image_name_list += [os.path.join(self.data_dir3, f) for f in os.listdir(self.data_dir3) if f.endswith('.jpg') or f.endswith('.png')]
    def __len__(self):
        return len(self.image_name_list)
    def get_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image
    def __getitem__(self, index):
        image_path = self.image_name_list[index]
        image = self.get_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        image_name = os.path.basename(image_path)
        return image, image_name

class CUSTOM_Dataset_For_Making_CAM(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_name_list = []
        self.transform = transforms.Compose([
            # transforms.Resize((512, 512)),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        for i in os.listdir(self.data_dir):
            if i.endswith('.jpg') or i.endswith('.png'):
                self.image_name_list.append(i)
    def __len__(self):
        return len(self.image_name_list)
    def get_image(self, image_name):
        image = Image.open(self.data_dir + image_name).convert('RGB')
        return image
    def __getitem__(self, index):
        image_name = self.image_name_list[index]
        image = self.get_image(image_name)

        # 检查图像尺寸并进行缩放 针对plantseg增加的限制
        width, height = image.size
        if width > 1024 or height > 1024:
            new_width = min(width, 1024)
            new_height = min(height, 1024)
            image = image.resize((new_width, new_height), Image.LANCZOS)
    
        if self.transform is not None:
            image = self.transform(image)
        return image, image_name

class ECSSD_Dataset_For_Making_CAM(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_name_list = []
        for i in os.listdir(self.data_dir):
            if i.endswith('.jpg') or i.endswith('.png'):
                self.image_name_list.append(i)
    def __len__(self):
        return len(self.image_name_list)
    def get_image(self, image_name):
        image = Image.open(self.data_dir + image_name).convert('RGB')
        return image
    def __getitem__(self, index):
        image_name = self.image_name_list[index]
        image = self.get_image(image_name)
        return image, image_name
    
# ==========================PASCAL VOC2012=========================================================
# 基础的VOC数据集类，每个都需要调用
class VOC_Dataset(torch.utils.data.Dataset):
    '''基础的VOC数据集类，用于加载图像、标签和掩码等信息。'''
    def __init__(self, root_dir, domain, with_id=False, with_tags=False, with_mask=False):
        self.root_dir = root_dir

        self.image_dir = self.root_dir + 'JPEGImages/'  # 17125张RGB原图
        self.xml_dir = self.root_dir + 'Annotations/'   # 17125个xml标注文件 
        self.mask_dir = self.root_dir + 'SegmentationClass/' # 2913个掩码图片 
        
        self.image_id_list = [image_id.strip() for image_id in open('./data/%s.txt'%domain).readlines()]
        
        self.with_id = with_id
        self.with_tags = with_tags
        self.with_mask = with_mask

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def get_tags(self, image_id):
        _, tags = read_xml(self.xml_dir + image_id + '.xml')
        return tags
    
    def __getitem__(self, index):
        image_id = self.image_id_list[index]

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        if self.with_tags:
            data_list.append(self.get_tags(image_id))

        if self.with_mask:
            data_list.append(self.get_mask(image_id))
        
        return data_list

# 训练CCAM时使用
class VOC_Dataset_For_Classification(VOC_Dataset):
    '''用于图像分类任务的VOC数据集子类。'''
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_tags=True)
        self.transform = transform
        data = read_json('./data/VOC_2012.json')
        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, tags = super().__getitem__(index)
        if self.transform is not None:
            image = self.transform(image)
        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, label

class VOC_Dataset_For_Segmentation(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        return image, mask

class VOC_Dataset_For_Evaluation(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_id=True, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    def __getitem__(self, index):
        image, image_id, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        return image, image_id, mask

class VOC_Dataset_For_WSSS(VOC_Dataset):
    def __init__(self, root_dir, domain, pred_dir, transform=None):
        super().__init__(root_dir, domain, with_id=True)
        self.pred_dir = pred_dir
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
    
    def __getitem__(self, index):
        image, image_id = super().__getitem__(index)
        mask = Image.open(self.pred_dir + image_id + '.png')
        
        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        return image, mask

class VOC_Dataset_For_Testing_CAM(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_tags=True, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
        
        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, tags, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, label, mask

# 提取CCAM和背景提示background cues时使用
class VOC_Dataset_For_Making_CAM(VOC_Dataset):
    def __init__(self, root_dir, domain):
        super().__init__(root_dir, domain, with_id=True, with_tags=True, with_mask=True)

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
        
        data = read_json('./data/VOC_2012.json')

        self.class_names = np.asarray(class_names[1:21])
        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, image_id, tags, mask = super().__getitem__(index)

        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, image_id, label, mask
