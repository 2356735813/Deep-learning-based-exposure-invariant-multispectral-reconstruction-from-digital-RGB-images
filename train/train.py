import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime
# 参数解析器
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--data_root", type=str, default='../dataset/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration*opt.end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# model
# 从命令行参数中获取预训练模型的路径
pretrained_model_path = opt.pretrained_model_path
# 从命令行参数中获取模型训练方法的名称
method = opt.method
# 根据指定的训练方法和预训练模型路径，创建一个神经网络模型，并将其移动到CUDA设备上（如果可用）
model = model_generator(method, pretrained_model_path).cuda()
# 计算并输出模型的参数数量。这行代码通过遍历模型的参数，并对每个参数的元素数量（参数的大小）进行求和，从而得到整个模型的参数数量
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# output path 生成一个带有日期和时间信息的输出目录路径
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
# 优化器和学习率调整器
optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging 保存训练过程中记录和保存关键信息
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

# Resume
# 从命令行参数中获取之前训练状态的保存路径
resume_file = opt.pretrained_model_path
# 如果存在之前的训练状态保存文件
if resume_file is not None:
    # 检查之前的训练状态保存文件是否存在。
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        # 加载之前保存的训练状态字典
        checkpoint = torch.load(resume_file)
        # 获取之前训练状态中记录的起始 epoch
        start_epoch = checkpoint['epoch']
        # 获取之前训练状态中记录的当前迭代次数
        iteration = checkpoint['iter']
        # 将之前保存的模型参数加载到当前模型中
        model.load_state_dict(checkpoint['state_dict'])
        # 将之前保存的优化器状态加载到当前优化器中
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    # 启用cuDNN的自动调优功能，以优化训练速度
    cudnn.benchmark = True
    # 初始化迭代计数器
    iteration = 0
    # 初始化记录最佳MRAE损失值
    record_mrae_loss = 1000

    while iteration<total_iteration:
        # 训练模式
        model.train()
        # 在训练过程中计算和记录损失值的平均值
        losses = AverageMeter()
        # 数据批量加载
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            # 在进行反向传播和优化之前，将优化器中的模型参数梯度清0，避免梯度积累问题
            optimizer.zero_grad()
            # 向前传播，得到模型预测output
            output = model(images)
            output = output
            # 计算损失
            loss = criterion_mrae(output, labels)
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数，调整学习率
            optimizer.step()
            scheduler.step()
            # 更新损失值，迭代数
            losses.update(loss.data)
            iteration = iteration+1
            if iteration % 20 == 0:
                # 输出当前学习率和损失
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                      % (iteration, total_iteration, lr, losses.avg))
            if iteration % 1000 == 0:
                # 使用验证数据评估性能，输出MRAE，RMSE,PSNR
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
                # Save model
                if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    # 保存模型的消息，显示保存的路径
                    save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss
                # print loss
                # 输出进展信息
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
                # 将相同的信息记录到日志文件中，，以便在训练过程中保留详细的训练日志
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
    return 0

# Validate
def validate(val_loader, model):
    # 评估模式
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

if __name__ == '__main__':
    main()
    print(torch.__version__)