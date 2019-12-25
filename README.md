将代码中`torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))` 改为`torch.save(net.module, '%s/net_%03d.pth' % (args.outf, epoch + 1))`即可解决 报错`'collections.OrderedDict' object has no attribute 'eval'`。源文件已经修改
# “华为云杯”2019人工智能创新应用大赛
比赛链接点击这里：[“华为云杯”2019人工智能创新应用大赛](https://competition.huaweicloud.com/information/1000021526/circumstances)


直接运行`src/train.py`文件即可
## 加注意力机制：1.[pytorch中加入注意力机制（CBMA）](https://blog.csdn.net/qq_38410428/article/details/103694759)。2.[keras, TensorFlow中加入注意力机制](https://blog.csdn.net/qq_38410428/article/details/103695032)

# 知乎详解

[知乎详解点击这里](https://zhuanlan.zhihu.com/p/98740628)

## 此代码为线下运行代码，不支持华为云端运行
