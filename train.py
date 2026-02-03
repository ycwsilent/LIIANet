import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from ycw.projects.aTemp.LIIANet.net.LIIANet import LIIANet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime

# 设置gpu显卡
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

opt = option().parse_args()

def seed_torch():                                                       # 随机种子设置,不太合理
    seed = random.randint(1, 1000000)                                   # 像这个每次都可能设置不一样的随机种子，感觉就像是在试 - 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("seed-",seed)
    # return seed                                                         # 返回 保存每次的随机种子
    
def train_init():                                                       # GPU训练初始化
    seed_torch()                    
    cudnn.benchmark = True                                              # 启用cudnn自动优化器
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    # return x
    
def train(epoch):                                                       # 执行一个完整的训练周期 - 一个Epoch
    model.train()                                                       # 模型设置为训练模式
    loss_print = 0                                                      # 该轮的损失(初始值为0) - 需要计算累计损失值
    pic_cnt = 0                                                         # 要处理的图片数量(一轮) - 将要返回这个值
    loss_last_10 = 0                                                    # 最近10个批次的损失和图片数量 -- 其实感觉和上面两个变量一样
    pic_last_10 = 0
    train_len = len(training_data_loader)                               # 训练数据集长度
    iter = 0                                                            # 该轮的迭代次数
    torch.autograd.set_detect_anomaly(opt.grad_detect)                  # 梯度异常检测
    for batch in tqdm(training_data_loader):                            # 迭代器
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3] # 输入图像low , 真值图像normal
        im1 = im1.cuda()                                                # 转移到gpu里面
        im2 = im2.cuda()
        
        # 使用随机伽马函数.提高泛化能力,默认是不使用,直接使用原图像
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = random.randint(opt.start_gamma,opt.end_gamma) / 100.0
            output_rgb_mid,output_rgb = model(im1 ** gamma)  
        else:
            output_rgb_mid,output_rgb = model(im1)                                     # 将low图像输入到模型中得到输出图像 -- 这里就是一个输出(最终的);
            # 后面要加中间监督,可能就是这里开始处理了 ----------------------------------------------
            
        gt_rgb = im2                                                    # 真值图像
        # output_hvi = model.HVIT(output_rgb)                             # 将最终的增强结果图像转到HVI空间中
        # gt_hvi = model.HVIT(gt_rgb)                                     # 将标签的真值图像转到HVI空间中

        output_hsv = model.HSVT(output_rgb)                             # 将最终的增强结果图像转到HVI空间中
        gt_hsv = model.HSVT(gt_rgb)   

        h,w = output_rgb_mid.shape[2:]                                  # 得到形状                                       # 
        gt_mid = torch.nn.functional.interpolate(gt_rgb, (h, w), mode='bicubic', align_corners=False)                   # 得到真值图下采样的
        # gt_mid = torch.nn.functional.interpolate(gt_rgb, (h, w), mode='bilinear', align_corners=False)   # 双线性插值                # 得到真值图下采样的
        # gt_hvi_mid = model.HVIT(gt_mid)                                 # 将真值图转到HVI
        # 添加中间损失计算              
        # HVI空间的mid损失              

        loss_mid = L1_loss(output_rgb_mid, gt_mid) + D_loss(output_rgb_mid, gt_mid) + E_loss(output_rgb_mid, gt_mid) + opt.P_weight * P_loss(output_rgb_mid, gt_mid)[0]

        # 开始计算损失 --- 分为两个部分
        # HVI空间的损失 --- 4种损失计算
        loss_hvi = L1_loss(output_hsv, gt_hsv) + D_loss(output_hsv, gt_hsv) + E_loss(output_hsv, gt_hsv) + opt.P_weight * P_loss(output_hsv, gt_hsv)[0]
        # RGB空间的损失 --- 4种损失计算 
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        # 其实就是RGB空间的损失占1,至于HVI空间所占多少可以控制,但是opt里面默认是1
        loss = loss_rgb + opt.HVI_weight * loss_hvi + loss_mid * opt.mid_hvi_weight          # 计算两个空间的损失,联合优化RGB以及HVI
        # loss = loss_rgb + opt.HVI_weight * loss_hvi           # 计算两个空间的损失,联合优化RGB以及HVI
        # loss = loss_rgb           # 计算两个空间的损失,联合优化RGB以及HVI
        # loss += loss_mid * opt.mid_hvi_weight 
        iter += 1                                                       # 一次迭代结束 --- 前向传播完成
        
        if opt.grad_clip:                                               # 进行梯度裁剪, opt里面默认true,需要裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()                                           # 清零梯度 
        loss.backward()                                                 # 开始反向传播 ---- 
        optimizer.step()                                                # 参数更新 利用学习率和反向传播的梯度来更新各；连接之前的权重
                                                                        # 
        loss_print = loss_print + loss.item()                           # 累加计算损失(该轮内有多张图片,就有多个iter)
        loss_last_10 = loss_last_10 + loss.item()                       # 累加10次iter内的损失 --- 其实就是该轮总损失
        pic_cnt += 1                                                    # 图片数量加1
        pic_last_10 += 1                                                # 10次的加1
        if iter == train_len:                                           # 如果迭代完最后一张图片时,一轮结束
                                                                        # 打印该轮平均学习率以及损失率
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                loss_last_10/pic_last_10, optimizer.param_groups[0]['lr']))
            loss_last_10 = 0
            pic_last_10 = 0
            # 保存最后一次迭代的图像 (train数据) --- 因为每次训练会将图片顺序打乱,因此这个图像很可能会变化(不是固定的某一张)
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            # 将输入图像也弄进去
            in_img = transforms.ToPILImage()((im1)[0].squeeze(0))
            # 将中间的图也输出出来看效果到底怎么样
            mid_img = transforms.ToPILImage()((output_rgb_mid)[0].squeeze(0))
            mid_gt = transforms.ToPILImage()((gt_mid)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            # 保存图像
            in_img.save(opt.val_folder+'training/in.png')
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
            mid_img.save(opt.val_folder+'training/mid.png')
            mid_gt.save(opt.val_folder+'training/mid_gt.png')
    return loss_print, pic_cnt                                         # 总损失,总数量(像v2-re数据集的train是689,如果设置batch=5,则此处pic_cnt=689//5=137)
                

def checkpoint(epoch):                              # 模型权重保存
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():                                # 数据集加载
    print('===> Loading datasets')
    if opt.lol_v1 or opt.lol_blur or opt.lolv2_real or opt.lolv2_syn or opt.SID or opt.SICE_mix or opt.SICE_grad:
        if opt.lol_v1:
            train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lol_blur:
            train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lol_blur)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        if opt.lolv2_real:
            train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_real)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.lolv2_syn:
            train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_lolv2_syn)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        
        if opt.SID:
            train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_eval_set(opt.data_val_SID)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_mix:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_grad:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    else:
        raise Exception("should choose a dataset")
    return training_data_loader, testing_data_loader

def build_model():                                  # 模型加载 默认从0开始训练,若需要继续训练或者使用什么预训练的,可以在opt设置(start_epoch)
    print('===> Building model ')
    model = LIIANet().cuda()                         # 加载模型        
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model

def make_scheduler():                               # 优化器选择 -- 待测试这个循环余弦退火
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)           # adam优化器,初始学习率opt设置,其他参数默认 
    if opt.cos_restart_cyclic:                      # 循环余弦退火策略
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:                           # 带热重启的余弦退火 -- 文中的选择是这个
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():                                    # 多目标损失函数,此处应着重考虑(点开细看)
    L1_weight   = opt.L1_weight                     # L1损失权重        - L1损失
    D_weight    = opt.D_weight                      # 结构相似性损失权重 - ssim损失
    E_weight    = opt.E_weight                      # 边缘损失权重      - Edge
    P_weight    = 1.0                               # 感知损失权重      - Lpips - 使用或者不使用这个就是 perc --- 这里直接设置1 感觉就没有用到前面的opt设置,
    # (后面发现上面train函数里面使用了opt.P_weight *),那就没有问题
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    return L1_loss,P_loss,E_loss,D_loss

if __name__ == '__main__':                          # 入口,主函数 
    
    '''
    preparision                                     # 准备
    '''
    train_init()                                    # 使用GPU加速
    training_data_loader, testing_data_loader = load_datasets()         # 使用数据集
    model = build_model()                           # 构建以及加载模型
    optimizer,scheduler = make_scheduler()          # 优化器以及,学习策略
    L1_loss,P_loss,E_loss,D_loss = init_loss()      # 初始化损失函数以及所占权重
    
    '''
    train                                           # 训练
    '''
    psnr = []                                       # 3个指标,峰值信噪比,结构相似性,图像相似度
    ssim = []
    lpips = []
    start_epoch=0                                   # 开始的轮数,若继续训练,这里应该也要变化
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          # 新建文件夹-存放结果 - results文件夹
        os.mkdir(opt.val_folder) 
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):       # 继续训练opt.nEpochs 这么多轮
        epoch_loss, pic_num = train(epoch)          # 执行训练 --- 这两个输出出来,到底起到了什么作用,未知
        scheduler.step()                            # 更新学习率
        
        if epoch % opt.snapshots == 0:              # 默认在opt里面设置10轮触发一次
            model_out_path = checkpoint(epoch)      # 模型保存
            norm_size = True                        # 可能是归一化尺寸

            # LOL three subsets                     # 对LOL的3个子集设置
            if opt.lol_v1:
                output_folder = 'LOLv1/'            # 输出的文件夹
                label_dir = opt.data_valgt_lol_v1   # 标签的文件夹 -- 下面类似
            if opt.lolv2_real:
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.lolv2_syn:
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.lol_blur:
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            if opt.SID:
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.SICE_mix:
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.SICE_grad:
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False

            im_dir = opt.val_folder + output_folder + '*.png'      # 将 文件保存 - 默认设置保存在 ./results/数据集名/*.png
            #                                                      # 进行验证 -- 这个还不清楚怎么弄的 - 为什么这里的alpha=0.8,eval里面是1(eval里面对应的re-psnr也是0.8)
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                 norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)
            
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)       # 用输出的文件与标签文件列表计算3个指标,默认不使用GT
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)                                # 得到至现在的总的3个指标的列表情况 
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)
        torch.cuda.empty_cache()                                # 清理GPU缓存
    
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")            # 写md文件,感觉像这里写的损失,只是opt文件里面的,其实P_weight似乎并没有使用(使用的本文件设置的固定值1)
    with open(f"./results/training/metrics{now}.md", "w") as f:
        # f.write("randSeed: "+ train_init() + "\n")              # 打印随机值
        f.write("dataset: "+ output_folder + "\n")  
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"HVI_weight: {opt.HVI_weight}\n")  
        f.write(f"L1_weight: {opt.L1_weight}\n")  
        f.write(f"D_weight: {opt.D_weight}\n")  
        f.write(f"E_weight: {opt.E_weight}\n")  
        f.write(f"P_weight: {opt.P_weight}\n")  
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
        for i in range(len(psnr)):
            f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | { psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n")  
        