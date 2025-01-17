import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
# from torch.utils.tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from core.model import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from utils import check_positive
from datetime import datetime
from skimage import filters

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='/media/omnisky/tld/Tian_Datasets/万张图/Original_Resized_Leaf_Foreground/', type=str)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)
parser.add_argument('--NewDisentangler', type=str, default='False')
parser.add_argument('--CBAM', type=str, default='False')

parser.add_argument('--Disentangle_spatial', type=str, default='False')
parser.add_argument('--Disentangle_cbam', type=str, default='False')
parser.add_argument('--Disentangle_Fca', type=str, default='False')

# Inference parameters
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--vis_dir', default='vis_cam', type=str)
parser.add_argument('--cam_png', type=str, default='False')

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--option', type=int, choices=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], 
                    help='Option to choose default data directory')
parser.add_argument('--model_num', default='0', type=str)
parser.add_argument('--threshold', default=0.25, type=str)
parser.add_argument('--crf_iteration', default=0, type=int)

if __name__ == '__main__':
    # Arguments
    args = parser.parse_args()

    # 根据用户选择设置默认路径
    if args.option == 1:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/农业/Publish_Dataset/Pixel-level_annotation/Image/'
    elif args.option == 2:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Original_Resized_Leaf_Foreground/'
    elif args.option == 3:
        args.data_dir = '/media/omnisky/tld/TianLuDa/CCAM/CUSTOM_Disease/experiments/predictions/Leaf_Foreground_filled/'
    elif args.option == 4:
        args.data_dir = '/media/omnisky/tld/TianLuDa/CCAM/CUSTOM_Disease/experiments/predictions/Leaf_Foreground_White/'
    elif args.option == 5:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Original_Resized_Leaf_Foreground_White/'
    elif args.option == 6:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Resized_Leaf_Foreground/'
    elif args.option == 7:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Resized_Leaf_Foreground_White/'
    elif args.option == 8:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Resized_picture/'
    elif args.option == 9:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Original_Resized_picture/'
    elif args.option == 10:
        args.data_dir = '/media/omnisky/tld/TianLuDa/CCAM/CUSTOM_Disease/experiments/predictions/Leaf_Foreground/'
    elif args.option == 11:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Leaf_Foreground_40/'    
    elif args.option == 12:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Original_Resized_Leaf_Foreground_40/'
    elif args.option == 13:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/CoSOD_dataset/VOC2012/'
    elif args.option == 14:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Extra_test_dataset/Resized_Image/'
    elif args.option == 15:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Extra_Original_Resized_Leaf/Original_picture/'
    elif args.option == 16:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Apple_Healthy/'
    elif args.option == 17:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/SOD_dataset/ECSSD/images/'
    elif args.option == 18:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/农业/Segmentation_Dataset/LeafSpot/train_img/'
    elif args.option == 19:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/农业/Segmentation_Dataset/LeafSpot/val_img/'
    elif args.option == 20:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/农业/plantseg/images/train/'
    elif args.option == 21:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/农业/plantseg/images/test/'
    else:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Original_Resized_Leaf_Foreground/'

    experiment_name = args.tag
    experiment_name += '@scale=%s'%args.scales
    model_path=args.model_num
    pred_dir = create_directory(f'./experiments/predictions_{model_path}/{experiment_name}/')
    if args.cam_png=='False':
        cam_path = create_directory(f'{args.vis_dir}_{model_path}/{experiment_name}')
    else:
        cam_path = create_directory(f'{args.vis_dir}_{model_path}/{experiment_name}/OriCAm')
    pred_crf_dir = create_directory(f'./experiments/predictions_{model_path}/{experiment_name}@t={args.threshold}@ccam_inference_crf={args.crf_iteration}/')
    model_path = './experiments/models/' + f'{args.tag}.pth.'+str(model_path)
    print(model_path)
    scales = [float(scale) for scale in args.scales.split(',')]   #[2,3]

    set_seed(args.seed)
    log_path = './experiments/logs/'+'{}.txt'.format(args.tag)
    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format('inference_CCAM'))
    
    # Transform, Dataset, DataLoader
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    if args.option == 17:
        dataset = ECSSD_Dataset_For_Making_CAM(args.data_dir)
    elif args.option != 13:
        dataset = CUSTOM_Dataset_For_Making_CAM(args.data_dir)
    else:
        dataset = VOC_Dataset_For_Making_CAM(args.data_dir, 'val')
    
    # Network
    model = get_model('detco',NewDisentangler=args.NewDisentangler,CBAM=args.CBAM,
                      Disentangle_spatial=args.Disentangle_spatial,Disentangle_cbam=args.Disentangle_cbam,
                      Disentangle_Fca=args.Disentangle_Fca)
    torch.cuda.empty_cache() #释放未使用的显存
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'
    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    ckpt = torch.load(model_path)
    flag = ckpt['flag']
    if the_number_of_gpu > 1:
        model.module.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt['state_dict'])

    # Evaluation
    model.eval()
    current_time = datetime.now()
    log_func('[i] Current time is {}'.format(current_time))
    eval_timer = Timer()
    eval_timer.tik()
    
    def get_cam(ori_image, scale):
        # preprocessing
        image = copy.deepcopy(ori_image)
        image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.BICUBIC)
        image = normalize_fn(image)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        flipped_image = image.flip(-1)
        images = torch.stack([image, flipped_image])

        images = images.cuda()
        
        # inferenece
        if args.NewDisentangler=='False':
            _, _, cams = model(images, inference=True)  # 通过模型进行推断，获取CAM
        elif args.NewDisentangler=='True':
            _, _, cams , _, _ = model(images, inference=True)  # 通过模型进行推断，获取CAM
        else:
            _, _, cams , _ = model(images, inference=True)  # 通过模型进行推断，获取CAM
        if flag:
            cams = 1 - cams

        # postprocessing
        cams = F.relu(cams)
        cams = cams[0] + cams[1].flip(-1)
        return cams

    vis_cam = True
    with torch.no_grad():
        length = len(dataset)
        for step, data in enumerate(dataset):
            ori_image, image_id=data[:2]
            ori_w, ori_h = ori_image.size
            label = np.array([1])
            if image_id.endswith('.jpg'):
                image_id = image_id.split(".jpg")[0]
            else:
                image_id = image_id.split(".png")[0]
            npy_path = pred_dir + image_id + '.npy'
            png_path = pred_crf_dir + image_id + '.png'
            # 获得图像的下采样和上采样的尺寸
            strided_size = get_strided_size((ori_h, ori_w), 4)  #[512, 512]-->[128, 128]
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)  #[512, 512]-->[512, 512]
            # 获取不同尺度下的正反两个 CAM
            cams_list = [get_cam(ori_image, scale) for scale in scales]  
            # 处理和融合 CAM，得到strided_cams 和 hr_cams。这两者分别表示低分辨率和高分辨率下的 CAM。
            strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
            strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)
            hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
            hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]
            # 获取标签中非零元素的索引
            keys = torch.nonzero(torch.from_numpy(label))[:, 0]
            # 根据索引提取相关的 CAM 并进行归一化
            strided_cams = strided_cams[keys]
            strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5
            hr_cams = hr_cams[keys]
            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5
            
            # 生成最终的 CAM 并叠加到原始图像上
            cam = torch.sum(hr_cams, dim=0)
            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = make_cam(cam).squeeze()
            cam = get_numpy_from_tensor(cam)

            image = np.array(ori_image)
            h, w, c = image.shape
            cam = (cam * 255).astype(np.uint8)
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
            
            if args.cam_png=='False':
                cam = colormap(cam)
                image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
                cv2.imwrite(f'{cam_path}/{image_id}.png', image.astype(np.uint8))
            else:
                image = cv2.addWeighted(0, 0, cam, 1, 0)
                cv2.imwrite(f'{cam_path}/{image_id}.png', image.astype(np.uint8))
            
            # 保存生成的 CAM 和相关信息
            keys = np.pad(keys + 1, (1, 0), mode='constant')
            np.save(npy_path, {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()})

            cams=hr_cams.cpu().numpy()
            # 对类激活图进行处理
            if args.threshold =='otsu':
                threshold_value = filters.threshold_otsu(cams)
                print("\nthreshold_value",threshold_value)
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold_value)
            else:
                threshold_value = args.threshold
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold_value)
            cams = np.argmax(cams, axis=0)
            # CRF（条件随机场）推理
            if args.crf_iteration > 0:
                cams = crf_inference_label(np.asarray(ori_image), cams, n_labels=keys.shape[0], t=args.crf_iteration)
            # 保存结果图像
            imageio.imwrite(png_path, (cams*255).astype(np.uint8))
            # log_func('[i] {}'.format(image_id))
            # torch.cuda.empty_cache() #释放未使用的显存
    current_time = datetime.now()
    log_func('[i] Current time is {}'.format(current_time))
    print("python3 inference_crf.py --experiment_name {} --domain {}".format(experiment_name, args.domain))