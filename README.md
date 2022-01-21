利用pytorch实现图像分类，其中包含的resnextefficientnet等图像分类网络

参考的git原文：https://github.com/lxztju/pytorch_classification

## 实现功能
* 基础功能利用pytorch实现图像分类
* 包含带有warmup的cosine学习率调整
* warmup的step学习率优调整
* 添加label smooth的pytorch实现（标签平滑）
* 可视化特征层

## 运行环境
* python3.7
* pytorch 1.1
* torchvision 0.3.0

## 代码仓库的使用

### 数据集形式
原始数据集存储形式为，同个类别的图像存储在同一个文件夹下，所有类别的图像存储在一个主文件夹data下。

```
|-- data
    |-- train
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label    
            |--*.jpg
        ...

    |-- val
        |--*.jpg
```

利用preprocess.py将数据集格式进行转换（图片格式不一样，更改图片格式，追加写入处理后的文件中）

```
python ./data/preprocess.py
```

转换后的数据集为，将训练集的路径与类别存储在train.txt文件中，测试机存储在val.txt中.
存储在处理后的数据集里面processed-data

```
# oil_train.txt
# oil_val.txt
# oil_test.txt(少量用于测试)
# oil_all_test.txt(所有的预测样本)

```


### 模型介绍
仓库中模型mobilenet,resnext模型来自于torchvision

efficientnet来自于 https://github.com/lukemelas/EfficientNet-PyTorch

### 训练

* 权重介绍

  weight文件夹：对应下载原网络训练的网络参数

  weightsresnext101_32x32d文件夹：表示训练模型对应保存本数据的模型参数

* 在`cfg.py`中修改合适的参数，并在train.py中选择合适的模型

```python
##数据集的类别
NUM_CLASSES = 14

#训练时batch的大小
BATCH_SIZE = 16

#网络默认输入图像的大小
INPUT_SIZE = 300
#训练最多的epoch
MAX_EPOCH = 20
# 使用gpu的数目
GPUS = 1
# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 0

WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# 初始学习率
LR = 1e-3
# 训练好模型的保存位置
SAVE_FOLDER = './weights'

# 采用的模型名称
model_name = 'resnext101_32x32d'
model_name = "efficientnet-b8"
```

1. 直接利用训练数据集进行训练
```shell
python train.py
```

2. 在训练的时候使用验证集，得到验证集合的准确率（这个要求gpu显存比较高）
```shell
python train_val.py
```

### 预测

在cfg.py中`TRAINED_MODEL`参数修改为指定的权重文件存储位置,在predict文件中可以选定是否使用tta

```shell
python predict.py
```

### 预测结果

预测结果是将图片的路径和lable存入到result中，存放的形式路径+标签

move_result.py是将result结果的.csv文件存储的图片路径根据分出类别进行创建新的文件夹将分类好的结果移动到对应标签里面

### 其他文件介绍

picture_show.ipynb：查看数据集中的图片，每一个类别展示5张照片

### 画图

#### TSNE作图

1.在draw_image文件夹中TSNE_draw，直接运行tsne.py，相关联的程序petrolem_dataset.py(这里设置对应分类类别的颜色，按照自己的标签进行设置），resnet.py对应做图片映射使用的网络，这里使用的resNet101网络。

借鉴文献：https://learnopencv.com/t-sne-for-feature-visualization/

分类的结果图片展示：

<img src="C:\Users\yanyi\Desktop\新建文件夹\Snipaste_2022-01-18_10-03-14.png" style="zoom:60%;" />

<img src="C:\Users\yanyi\Desktop\新建文件夹\Snipaste_2022-01-18_10-03-55.png" style="zoom: 80%;" />



#### UMAP作图

1.在draw_image文件夹中UMAP_draw，直接运行umap_show.ipynb，相关联的程序petrolem_dataset.py，resnet.py对应做图片映射使用的网络，这里使用的resNet101网络。

下面是umap的效果：

<img src="C:\Users\yanyi\Desktop\新建文件夹\73d89e8c317dc811316275d1bdac7fc.png" style="zoom:80%;" />

​        下面来介绍一下TSNE和UMAP的差别，首先，在高维空间内，描述两个点（两个张量）的距离不一样，tSNE采取的是“概率算法”，即把两个点的距离转换成概率，若 i 与 j 这两个点距离比较近的话，它所对应的概率较大；而UMAP采取的是传统的欧式距离。

​         在做图像分类结果进行展示，两种方式作图提取的特征都是使用的resnet做的特征提取，区别点就是在于两者的降为和点雨点之间的比较的差异。

相关介绍的原理：(https://blog.csdn.net/weixin_33295562/article/details/113029121?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-1-113029121.pc_agg_new_rank&utm_term=UMAP