# 前言

#### 垃圾分类的华为云提交版本点击这里，详细版本：https://github.com/wusaifei/garbage_classify

1.图像分类的更多tricks（注意力机制 keras，TensorFlow和pytorch 版本等）：[图像分类比赛tricks：“华为云杯”2019人工智能创新应用大赛](https://zhuanlan.zhihu.com/p/98740628)

2.大家如果对目标检测比赛比较感兴趣的话，可以看一下我这篇对目标检测比赛tricks的详细介绍：[目标检测比赛中的tricks（已更新更多代码解析)](https://zhuanlan.zhihu.com/p/102817180)

3.目标检测比赛笔记：[目标检测比赛笔记](https://zhuanlan.zhihu.com/p/137567177)

4.如果对换脸技术比较感兴趣的同学可以点击这里：[deepfakes/faceswap：换脸技术详细教程，手把手教学，简单快速上手！！](https://zhuanlan.zhihu.com/p/376853800)

5.在日常调参的摸爬滚打中，参考了不少他人的调参经验，也积累了自己的一些有效调参方法，慢慢总结整理如下。希望对新晋算法工程师有所助力呀～：[写给新手炼丹师：2021版调参上分手册](https://zhuanlan.zhihu.com/p/376068083)

6.[深度学习中不同类型卷积的综合介绍：2D卷积、3D卷积、转置卷积、扩张卷积、可分离卷积、扁平卷积、分组卷积、随机分组卷积、逐点分组卷积等](https://zhuanlan.zhihu.com/p/366744794)

7.分类必备知识:[Softmax函数和Sigmoid函数的区别与联系](https://zhuanlan.zhihu.com/p/356976844)、[深度学习中学习率和batchsize对模型准确率的影响](https://zhuanlan.zhihu.com/p/277487038)、[准确率(Precision)、召回率(Recall)、F值(F-Measure)、平均正确率，IoU](https://zhuanlan.zhihu.com/p/101101207)、[利用python一层一层可视化卷积神经网络，以ResNet50为例](https://zhuanlan.zhihu.com/p/101038013)

8.[pytorch笔记：Efficientnet微调](https://zhuanlan.zhihu.com/p/102467338)

9.[keras, TensorFlow中加入注意力机制](https://zhuanlan.zhihu.com/p/99260231)、[pytorch中加入注意力机制（CBAM），以ResNet为例。解析到底要不要用ImageNet预训练？如何加预训练参数？](https://zhuanlan.zhihu.com/p/99261200)


# “华为云杯”2019人工智能创新应用大赛

## 赛题背景

比赛链接：[华为云人工智能大赛·垃圾分类挑战杯](https://developer.huaweicloud.com/competition/competitions/1000007620/introduction)

如今，垃圾分类已成为社会热点话题。其实在2019年4月26日，我国住房和城乡建设部等部门就发布了《关于在全国地级及以上城市全面开展生活垃圾分类工作的通知》，决定自2019年起在全国地级及以上城市全面启动生活垃圾分类工作。到2020年底，46个重点城市基本建成生活垃圾分类处理系统。

人工垃圾分类投放是垃圾处理的第一环节，但能够处理海量垃圾的环节是垃圾处理厂。然而，目前国内的垃圾处理厂基本都是采用人工流水线分拣的方式进行垃圾分拣，存在工作环境恶劣、劳动强度大、分拣效率低等缺点。在海量垃圾面前，人工分拣只能分拣出极有限的一部分可回收垃圾和有害垃圾，绝大多数垃圾只能进行填埋，带来了极大的资源浪费和环境污染危险。

随着深度学习技术在视觉领域的应用和发展，让我们看到了利用AI来自动进行垃圾分类的可能，通过摄像头拍摄垃圾图片，检测图片中垃圾的类别，从而可以让机器自动进行垃圾分拣，极大地提高垃圾分拣效率。

因此，华为云面向社会各界精英人士举办了本次垃圾分类竞赛，希望共同探索垃圾分类的AI技术，为垃圾分类这个利国利民的国家大计贡献自己的一份智慧。

## 赛题说明
本赛题采用深圳市垃圾分类标准，赛题任务是对垃圾图片进行分类，即首先识别出垃圾图片中物品的类别（比如易拉罐、果皮等），然后查询垃圾分类规则，输出该垃圾图片中物品属于可回收物、厨余垃圾、有害垃圾和其他垃圾中的哪一种。
模型输出格式示例：
    
    {

        " result ": "可回收物/易拉罐"

    }

## 垃圾种类40类

    {
        "0": "其他垃圾/一次性快餐盒",
        "1": "其他垃圾/污损塑料",
        "2": "其他垃圾/烟蒂",
        "3": "其他垃圾/牙签",
        "4": "其他垃圾/破碎花盆及碟碗",
        "5": "其他垃圾/竹筷",
        "6": "厨余垃圾/剩饭剩菜",
        "7": "厨余垃圾/大骨头",
        "8": "厨余垃圾/水果果皮",
        "9": "厨余垃圾/水果果肉",
        "10": "厨余垃圾/茶叶渣",
        "11": "厨余垃圾/菜叶菜根",
        "12": "厨余垃圾/蛋壳",
        "13": "厨余垃圾/鱼骨",
        "14": "可回收物/充电宝",
        "15": "可回收物/包",
        "16": "可回收物/化妆品瓶",
        "17": "可回收物/塑料玩具",
        "18": "可回收物/塑料碗盆",
        "19": "可回收物/塑料衣架",
        "20": "可回收物/快递纸袋",
        "21": "可回收物/插头电线",
        "22": "可回收物/旧衣服",
        "23": "可回收物/易拉罐",
        "24": "可回收物/枕头",
        "25": "可回收物/毛绒玩具",
        "26": "可回收物/洗发水瓶",
        "27": "可回收物/玻璃杯",
        "28": "可回收物/皮鞋",
        "29": "可回收物/砧板",
        "30": "可回收物/纸板箱",
        "31": "可回收物/调料瓶",
        "32": "可回收物/酒瓶",
        "33": "可回收物/金属食品罐",
        "34": "可回收物/锅",
        "35": "可回收物/食用油桶",
        "36": "可回收物/饮料瓶",
        "37": "有害垃圾/干电池",
        "38": "有害垃圾/软膏",
        "39": "有害垃圾/过期药物"
    }
## efficientNet默认参数

        (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),

efficientNet的论文地址：https://arxiv.org/pdf/1905.11946.pdf

## 代码解析

直接运行`src/train.py`文件即可。

加注意力机制代码：

1. [pytorch中加入注意力机制（CBMA）](https://blog.csdn.net/qq_38410428/article/details/103694759)。

2. [keras, TensorFlow中加入注意力机制](https://blog.csdn.net/qq_38410428/article/details/103695032)。


## 前期准备
* 克隆此存储库
    
    
    
      git clone https://github.com/wusaifei/HWCC_image_classification.git
    

* [垃圾分类数据集下载地址，此链接已经不存在请用下面的百度云盘下载。](https://modelarts-competitions.obs.cn-north-1.myhuaweicloud.com/garbage_classify/dataset/garbage_classify.zip)

* 垃圾分类数据集下载地址链接：https://pan.baidu.com/s/11xp0jBKAitU8r0_RWVpX1Q ， 提取码：jqa1 

* 扩充数据集：链接：https://pan.baidu.com/s/1SulD2MqZx_U891JXeI2-2g ，提取码：epgs


# 知乎详解

[知乎详解点击这里](https://zhuanlan.zhihu.com/p/98740628)

##### 此代码为线下运行代码，不支持华为云端运行。

# 其他问题

将代码中`torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))` 改为`torch.save(net.module, '%s/net_%03d.pth' % (args.outf, epoch + 1))`即可解决报错`collections.OrderedDict' object has no attribute 'eval`，源文件已经修改。
