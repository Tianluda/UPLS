from datetime import datetime
import sys

import matplotlib

matplotlib.use('Agg')
from torchvision import transforms
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from utils import *
from core.datasets import *
from core.model import *
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from core.loss import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from shutil import copyfile
import matplotlib.pyplot as plt
from optimizer import PolyOptimizer
from fvcore.nn import FlopCountAnalysis, flop_count_table

os.environ["NUMEXPR_NUM_THREADS"] = "8"
parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='/media/omnisky/tld/TianLuDa/CCAM/CUSTOM_Disease/experiments/predictions/Leaf_Foreground/', type=str)
# 添加选项参数，用于选择默认路径
parser.add_argument('--option', type=int, choices=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], 
                    help='Option to choose default data directory')

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--NewDisentangler', type=str, default='False')
parser.add_argument('--CBAM', type=str, default='False')
parser.add_argument('--CA', type=str, default='False')
parser.add_argument('--constraint_term', type=str, default='False')

parser.add_argument('--Disentangle_spatial', type=str, default='False')
parser.add_argument('--Disentangle_cbam', type=str, default='False')
parser.add_argument('--Disentangle_Fca', type=str, default='False')
###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('--depth', default=50, type=int)

parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--print_ratio', default=0.2, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)

parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--pretrained', type=str, required=True,
                        help='adopt different pretrained parameters, [supervised, mocov2, detco, plant]')

flag = True

if __name__ == '__main__':
    # Arguments
    ###################################################################################
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
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Extre_Leaf_Foreground/'
    elif args.option == 16:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/万张图/Apple_Healthy/'
    elif args.option == 17:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/SOD_dataset/ECSSD/images/'
    elif args.option == 18:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/农业/Segmentation_Dataset/LeafSpot/train_img/'
    elif args.option == 19:
        args.data_dir = '/media/omnisky/tld/TianLuDa/CCAM/CUSTOM_new/experiments/predictions_best/CCAM_LeafSpot_MOCO_1_filled_Foreground/'
    elif args.option == 20:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/农业/plantseg/images/train/'
    elif args.option == 21:
        args.data_dir = '/media/omnisky/tld/Tian_Datasets/农业/plantseg/images/test/'
    else:
        args.data_dir = '/media/omnisky/tld/TianLuDa/CCAM/CUSTOM_Disease/experiments/predictions/Leaf_Foreground/'

    log_dir = create_directory('./experiments/logs/')
    data_dir = create_directory('./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory('./experiments/tensorboards/{}/'.format(args.tag))

    log_path = log_dir + '{}.txt'.format(args.tag)
    data_path = data_dir + '{}.json'.format(args.tag)
    model_path = model_dir + '{}.pth'.format(args.tag)
    cam_path = './experiments/images/{}'.format(args.tag)
    create_directory(cam_path)
    create_directory(cam_path + '/train')
    create_directory(cam_path + '/test')
    create_directory(cam_path + '/train/colormaps')
    create_directory(cam_path + '/test/colormaps')

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args.tag))
    # log_func()

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(0)

    if args.option != 13:
        # data augmentation
        train_transform = transforms.Compose([
            # the input size is decided by the adopted datasets
            transforms.Resize(size=(192, 192)),   #病害检测
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        train_dataset = CUSTOM_Dataset(args.data_dir, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    else:
        train_transform = transforms.Compose([
            transforms.Resize(size=(512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        train_dataset = VOC_Dataset_For_Classification(args.data_dir, 'train_aug', train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func('[i] #train data'.format(len(train_dataset)))
    # log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    ###################################################################################
    # Network
    ###################################################################################
    model = get_model(pretrained=args.pretrained,NewDisentangler=args.NewDisentangler,CBAM=args.CBAM,
                      Disentangle_spatial=args.Disentangle_spatial,Disentangle_cbam=args.Disentangle_cbam,
                      Disentangle_Fca=args.Disentangle_Fca)
    param_groups = model.get_parameter_groups()
    torch.cuda.empty_cache() #释放未使用的显存
    
    model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    def calculate_flops(model, input_size):
        input_tensor = torch.randn(*input_size).cuda()
        flop_count = FlopCountAnalysis(model, input_tensor)
        flops = flop_count.total()
        return flops

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)

    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    if args.NewDisentangler=='False':
        criterion = [SimMaxLoss(metric='cos', alpha=args.alpha).cuda(), SimMinLoss(metric='cos').cuda(),
                     SimMaxLoss(metric='cos', alpha=args.alpha).cuda()]
    elif args.NewDisentangler=='True':
        criterion = [SimMaxLoss(metric='cos', alpha=args.alpha).cuda(), SimMinLoss(metric='cos').cuda(),
                     SimMaxLoss(metric='cos', alpha=args.alpha).cuda(),
                     IncreaseFgDiffLoss().cuda(),DecreaseBgDiffLoss().cuda()]
    else:
        criterion = [SimMaxLoss(metric='cos', alpha=args.alpha).cuda(), SimMinLoss(metric='cos').cuda(),
                     SimMaxLoss(metric='cos', alpha=args.alpha).cuda(), DecreaseSimLoss(metric='cos').cuda(),
                     IncreaseSimLoss(metric='cos', alpha=args.alpha)
                     ]
        
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration)

    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train': [],
        'validation': []
    }
    current_time = datetime.now()
    print("Current time:", current_time)
    log_func('[i] Current time: {}'.format(current_time))

    train_timer = Timer()
    eval_timer = Timer()

    metrics = ['loss', 'positive_loss', 'negative_loss']
    if args.NewDisentangler in ['True', '1', '2','3','4','5','6','7']:
        metrics.append('FgDiff_Loss')
        metrics.append('BgDiff_Loss')
    if args.constraint_term == 'True':
        metrics.append('constraint_term')
    if args.constraint_term == 'Only':
        metrics = ['constraint_term']
    train_meter = Average_Meter(metrics)

    writer = SummaryWriter(tensorboard_dir)
    min_loss = float('inf')

    for epoch in range(args.max_epoch):
        for iteration, (images, _) in enumerate(train_loader):

            images = images.cuda()
            optimizer.zero_grad()

            if args.NewDisentangler=='False':
                fg_feats, bg_feats, ccam = model(images)
                loss1,sim1 = criterion[0](fg_feats)
                loss2,sim2 = criterion[1](bg_feats, fg_feats)
                loss3,sim3 = criterion[2](bg_feats)
                if args.constraint_term=='True':
                    constraint_term = torch.sigmoid(sim3 - sim2) - 0.5
                    constraint_term = constraint_term.mean()  # 将约束项的值取平均，确保它是一个标量
                    loss = loss1 + loss2 + loss3 + constraint_term
                elif args.constraint_term=='Only':
                    constraint_term = torch.sigmoid(sim3 - sim2) - 0.5
                    constraint_term = constraint_term.mean()  # 将约束项的值取平均，确保它是一个标量
                    loss = constraint_term
                else:
                    loss = loss1 + loss2 + loss3
            elif args.NewDisentangler=='True':
                fg_feats, bg_feats, ccam, diff_fg, diff_bg = model(images)
                loss1,sim1 = criterion[0](fg_feats)
                loss2,sim2 = criterion[1](bg_feats, fg_feats)
                loss3,sim3 = criterion[2](bg_feats)
                loss4 = criterion[3](diff_fg)
                loss5 = criterion[4](diff_bg)
                if args.constraint_term=='True':
                    constraint_term = torch.sigmoid(sim3 - sim2) - 0.5
                    constraint_term = constraint_term.mean()  # 将约束项的值取平均，确保它是一个标量
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + constraint_term
                elif args.constraint_term=='Only':
                    constraint_term = torch.sigmoid(sim3 - sim2) - 0.5
                    constraint_term = constraint_term.mean()
                    loss = constraint_term
                else:
                    loss = loss1 + loss2 + loss3 + loss4 + loss5
            else:
                fg_feats, bg_feats, ccam, global_feats = model(images)
                loss1,sim1 = criterion[0](fg_feats)
                loss2,sim2 = criterion[1](bg_feats, fg_feats)
                loss3,sim3 = criterion[2](bg_feats)
                loss4,sim4 = criterion[3](global_feats, fg_feats)
                loss5,sim5 = criterion[4](global_feats, bg_feats)
                if args.constraint_term=='True':
                    constraint_term = torch.sigmoid(sim3 - sim2) - 0.5
                    constraint_term = constraint_term.mean()  # 将约束项的值取平均，确保它是一个标量
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + constraint_term
                elif args.constraint_term=='Only':
                    constraint_term = torch.sigmoid(sim3 - sim2) - 0.5
                    constraint_term = constraint_term.mean()
                    loss = constraint_term
                else:
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 # 前景+背景+Global
            loss.backward()
            optimizer.step()

            if epoch == 0 and iteration == (len(train_loader)-1):
                flag = check_positive(ccam)
                print(f"Is Negative: {flag}")
            if flag:
                ccam = 1 - ccam

            if args.constraint_term=='Only':
                train_meter.add({'constraint_term':constraint_term.item()})
            elif args.NewDisentangler=='False':
                train_meter.add({
                    'loss': loss.item(),
                    'positive_loss': loss1.item() + loss3.item(),
                    'negative_loss': loss2.item(),
                })
                if args.constraint_term=='True':
                    train_meter.add({'constraint_term':constraint_term.item()})      
            elif args.NewDisentangler=='True':
                train_meter.add({
                    'loss': loss.item(),
                    'positive_loss': loss1.item() + loss3.item(),
                    'negative_loss': loss2.item(),
                    'FgDiff_Loss': loss4.item(),
                    'BgDiff_Loss': loss5.item(),
                })  
                if args.constraint_term=='True':
                    train_meter.add({'constraint_term':constraint_term.item()})      
            else:
                train_meter.add({
                    'loss': loss.item(),
                    'positive_loss': loss1.item() + loss3.item(),
                    'negative_loss': loss2.item(),
                    'FgDiff_Loss': loss4.item(),
                    'BgDiff_Loss': loss5.item(),
                })
                if args.constraint_term=='True':
                    train_meter.add({'constraint_term':constraint_term.item()})                  
            #################################################################################################
            # For Log
            #################################################################################################

            if (iteration + 1) % len(train_loader) == 0:
                visualize_heatmap(args.tag, images.clone().detach(), ccam, 0, iteration)
                if args.constraint_term=='Only':
                    constraint_term = train_meter.get(clear=True)
                elif args.NewDisentangler=='False' and args.constraint_term=='False':
                    loss, positive_loss, negative_loss = train_meter.get(clear=True)
                elif args.NewDisentangler=='False' and args.constraint_term=='True':
                    loss, positive_loss, negative_loss,constraint_term = train_meter.get(clear=True)
                elif args.NewDisentangler=='True' and args.constraint_term=='False':
                    loss, positive_loss, negative_loss, FgDiff_Loss, BgDiff_Loss= train_meter.get(clear=True)
                elif args.NewDisentangler=='True' and args.constraint_term=='True':
                    loss, positive_loss, negative_loss, FgDiff_Loss, BgDiff_Loss, constraint_term= train_meter.get(clear=True)
                elif args.constraint_term=='False':
                    loss, positive_loss, negative_loss, FgDiff_Loss, BgDiff_Loss= train_meter.get(clear=True)
                else:
                    loss, positive_loss, negative_loss, FgDiff_Loss, BgDiff_Loss, constraint_term= train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                if args.constraint_term=='Only':
                    data = {
                        'epoch': epoch,
                        'max_epoch': args.max_epoch,
                        'iteration': iteration + 1,
                        'learning_rate': learning_rate,
                        'loss': loss,
                        'constraint_term':constraint_term,
                        'time': train_timer.tok(clear=True),
                    }
                    data_dic['train'].append(data)
                    log_func('[i]\t'
                            'Epoch[{epoch:,}/{max_epoch:,}],\t'
                            'iteration={iteration:,}, \t'
                            'learning_rate={learning_rate:.4f}, \t'
                            'loss={loss:.4f}, \t'
                            'constraint_term={constraint_term:.4f}, \t'
                            'time={time:.0f}sec'.format(**data)
                            )
                elif args.NewDisentangler=='False' and args.constraint_term=='False':
                    data = {
                        'epoch': epoch,
                        'max_epoch': args.max_epoch,
                        'iteration': iteration + 1,
                        'learning_rate': learning_rate,
                        'loss': loss,
                        'positive_loss': positive_loss,
                        'negative_loss': negative_loss,
                        'time': train_timer.tok(clear=True),
                    }
                    data_dic['train'].append(data)
                    log_func('[i]\t'
                            'Epoch[{epoch:,}/{max_epoch:,}],\t'
                            'iteration={iteration:,}, \t'
                            'learning_rate={learning_rate:.6f}, \t'
                            'loss={loss:.4f}, \t'
                            'positive_loss={positive_loss:.4f}, \t'
                            'negative_loss={negative_loss:.4f}, \t'
                            'time={time:.0f}sec'.format(**data)
                            )
                elif args.NewDisentangler=='False' and args.constraint_term=='True':
                    data = {
                        'epoch': epoch,
                        'max_epoch': args.max_epoch,
                        'iteration': iteration + 1,
                        'learning_rate': learning_rate,
                        'loss': loss,
                        'positive_loss': positive_loss,
                        'negative_loss': negative_loss,
                        'constraint_term':constraint_term,
                        'time': train_timer.tok(clear=True),
                    }
                    data_dic['train'].append(data)
                    log_func('[i]\t'
                            'Epoch[{epoch:,}/{max_epoch:,}],\t'
                            'iteration={iteration:,}, \t'
                            'learning_rate={learning_rate:.6f}, \t'
                            'loss={loss:.4f}, \t'
                            'positive_loss={positive_loss:.4f}, \t'
                            'negative_loss={negative_loss:.4f}, \t'
                            'constraint_term={constraint_term:.4f}, \t'
                            'time={time:.0f}sec'.format(**data)
                            )                                    
                elif args.constraint_term=='False':
                    data = {
                        'epoch': epoch,
                        'max_epoch': args.max_epoch,
                        'iteration': iteration + 1,
                        'learning_rate': learning_rate,
                        'loss': loss,
                        'positive_loss': positive_loss,
                        'negative_loss': negative_loss,
                        'FgDiff_Loss': FgDiff_Loss,
                        'BgDiff_Loss': BgDiff_Loss,
                        'time': train_timer.tok(clear=True),
                    }
                    data_dic['train'].append(data)
                    log_func('[i]\t'
                            'Epoch[{epoch:,}/{max_epoch:,}],\t'
                            'iteration={iteration:,}, \t'
                            'learning_rate={learning_rate:.6f}, \t'
                            'loss={loss:.4f}, \t'
                            'positive_loss={positive_loss:.4f}, \t'
                            'negative_loss={negative_loss:.4f}, \t'
                            'FgDiff_Loss={FgDiff_Loss:.4f}, \t'
                            'BgDiff_Loss={BgDiff_Loss:.4f}, \t'
                            'time={time:.0f}sec'.format(**data)
                            )
                else:
                    data = {
                        'epoch': epoch,
                        'max_epoch': args.max_epoch,
                        'iteration': iteration + 1,
                        'learning_rate': learning_rate,
                        'loss': loss,
                        'positive_loss': positive_loss,
                        'negative_loss': negative_loss,
                        'FgDiff_Loss': FgDiff_Loss,
                        'BgDiff_Loss': BgDiff_Loss,
                        'constraint_term':constraint_term,
                        'time': train_timer.tok(clear=True),
                    }
                    data_dic['train'].append(data)
                    log_func('[i]\t'
                            'Epoch[{epoch:,}/{max_epoch:,}],\t'
                            'iteration={iteration:,}, \t'
                            'learning_rate={learning_rate:.6f}, \t'
                            'loss={loss:.4f}, \t'
                            'positive_loss={positive_loss:.4f}, \t'
                            'negative_loss={negative_loss:.4f}, \t'
                            'FgDiff_Loss={FgDiff_Loss:.4f}, \t'
                            'BgDiff_Loss={BgDiff_Loss:.4f}, \t'
                            'constraint_term={constraint_term:.4f}, \t'
                            'time={time:.0f}sec'.format(**data)
                            )
                    
                writer.add_scalar('Train/loss', loss, iteration)
                writer.add_scalar('Train/learning_rate', learning_rate, iteration)
                # break
        #################################################################################################
        # Evaluation
        #################################################################################################
        # save_model_fn()
        if epoch<=4 or (epoch+1) % 10 == 0:
        # if epoch<=19 or (epoch+1) % 10 == 0:
            torch.save({'state_dict': model.module.state_dict() if (the_number_of_gpu > 1) else model.state_dict(),
                        'flag': flag}, model_path+'.'+str(epoch))
        log_func('[i] save model')
        if loss < min_loss:
            min_loss = loss
            # Save the model checkpoint
            torch.save({'state_dict': model.module.state_dict() if (the_number_of_gpu > 1) else model.state_dict(),
                        'flag': flag}, model_path + '.best')
            log_func(f"New best model saved at epoch {epoch} with loss {min_loss:.4f}")

    current_time = datetime.now()
    print("Current time:", current_time)
    log_func('[i] Current time: {}'.format(current_time))
    print(args.tag)
