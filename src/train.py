# -*- coding: utf-8 -*-
from torchvision import datasets, transforms
import torch
from torch import nn
import torch.optim as optim
import argparse
import warnings
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式
from efficientnet_pytorch import EfficientNet
from label_smooth import LabelSmoothSoftmaxCE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

warnings.filterwarnings("ignore")

#   to the ImageFolder structure
data_dir = "../data/"

# Number of classes in the dataset
num_classes = 54

# Batch size for training (change depending on how much memory you have)
batch_size = 20  # 批处理尺寸(batch_size)

# Number of epochs to train for
EPOCH = 150

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
# feature_extract = True
feature_extract = False
# 超参数设置
pre_epoch = 0  # 定义已经遍历数据集的次数


def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据


input_size = 380

# 用Adam优化器
net = EfficientNet.from_pretrained('efficientnet-b4')
net._fc.out_features = num_classes


# 显示网络信息
print(net)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练使用多GPU，测试单GPU
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

net.to(device)

# 读取网络信息
# net.load_state_dict(torch.load('./model/net_035.pth'))

# Send the model to GPU
net = net.to(device)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ]),

    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=12,
                                   collate_fn=my_collate_fn) for x in ['train', 'val']}
b = image_datasets['train'].class_to_idx

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch DeepNetwork Training')
parser.add_argument('--outf', default='./model/model', help='folder to output images and model checkpoints')  # 输出结果保存路径

args = parser.parse_args()
params_to_update = net.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            print("\t", name)


def main():
    ii = 0
    LR = 1e-3  # 学习率
    best_acc = 0  # 初始化best test accuracy
    print("Start Training, DeepNetwork!")  # 定义遍历数据集的次数


    # criterion
    criterion = LabelSmoothSoftmaxCE()

    # optimizer
    optimizer = optim.Adam(params_to_update, lr=LR, betas=(0.9, 0.999), eps=1e-9)


    # scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

    with open("./txt/acc.txt", "w") as f:
        with open("./txt/log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                # scheduler.step(epoch)

                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0

                for i, data in enumerate(dataloaders_dict['train'], 0):
                    # 准备数据
                    length = len(dataloaders_dict['train'])

                    input, target = data
                    input, target = input.to(device), target.to(device)

                    # 训练
                    optimizer.zero_grad()
                    # forward + backward
                    output = net(input)
                    loss = criterion(output, target)

                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += predicted.eq(target.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                             100. * float(correct) / float(total)))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                100. * float(correct) / float(total)))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in dataloaders_dict['val']:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).cpu().sum()
                    print('测试分类准确率为：%.3f%%' % (100. * float(correct) / float(total)))
                    acc = 100. * float(correct) / float(total)
                    scheduler.step(acc)

                    # 将每次测试结果实时写入acc.txt文件中
                    if (ii % 1 == 0):
                        print('Saving model......')
                        torch.save(net.module, '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("./txt/best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)


if __name__ == "__main__":
    main()
