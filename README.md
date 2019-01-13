# deeplabv3_pytorch
deeplab v3 implement in pytorch

## 环境要求
pytorch==1.0.0<br>
torchvision=0.2.1<br>
torchsummaryX(可选)<br>
tensorboardX （可选）<br>
glob<br>
imageio<br>
其中torchsummaryX的功能和keras中model.summary()一样，可以打印出来model的结构，查看结构是否正确。安装方式如下：<br>
```
pip install torchsummaryX
```
tensorboardX的功能类似tensorflow中的checkpoint保存训练过程，可用于训练过程的可视化。
```
pip install tensorboardX
```
[使用方法参考](https://github.com/nmhkahn/torchsummaryX）
另外介绍几个相关的可视化工具，torchsummary(类似torchsummaryX),[graphviz](https://github.com/szagoruyko/pytorchviz)<br>
## 用法
文件结构<br>
```
deeplabv3_pytorch.py   #模型定义
DataLoader.py          #数据加载
train.py               #模型训练
test.py                #测试/验证
```
### useage 1
在模型定义文件（deeplabv3_pytorch.py ）中修改最后的部分：<br>
```
if __name__=="__main__":
    model_framework()
    #_main()
```
可以看到网络结构，各个层的参数<br>
```
if __name__=="__main__":
    #model_framework()
    _main()
```
### useage 2
数据加载文件（DataLoader.py ）定义了加载的方式，主要做了归一化和转换为Tensor的变换。需要根据自己的数据进行修改，这里采用的数据集是BDD100k的数据格式。
```
#定义了样本和标签的路径
self.images_path = os.path.join(train_path,"train")
self.labels_path = os.path.join(label_path,"train")
```
```
#定义了样本和标签的对应的命名区别，根据样本来寻找对应的标签
self.temp_name = self.image_name.split(".")[0]+"_drivable_id"+".png"
```
### useage 3
在训练文件train.py中定义了模型训练的一些参数。
首先定义了模型的输入尺寸、类别数目、数据集的具体路径以及backbone（可以根据自己的需要进行修改），目前只有xception和mobilenet两种，其他backbone正在开发中。
```
input_shape=(720,1280) 
backbone="xception"
num_classes=3
batch=3
model = deeplab_v3(input_shape,backbone,num_classes)
train_path = "/home/deeplearning/code/BDD100K/dataset/bdd100k_images/images/100k"
label_path = "/home/deeplearning/code/BDD100K/dataset/bdd100k_drivable_maps/drivable_maps/labels"
```
接着定义了各种文件的保存路径，在文件夹checkpoint中。同时，采用了第三方库tensorboardX 将训练过程保存，用于tensorboard的可视化。最后的模型保存，只保存##了训练参数。注意这里我才用了多个gpu进行训练，[0,1,2]表示显卡编号，三块显卡1080ti。
```
model = torch.nn.DataParallel(model, device_ids=[0,1,2], output_device=None, dim=0)
from tensorboardX import SummaryWriter
torch.save(model.state_dict(), temp_path)
```
### useage 4
在测试/验证文件test.py中定义了模型测试的方式。<br>
主要关注模型加载的地方，由于前面训练是用到三块GPU，当你在测试时如果配置和训练的代码是一样的（类似训练中断后，加载之前保存的模型继续训练），不过不需要一系列反向传播的操作，那么就不需要考虑这里所说的问题。
但是往往在测试时只有cpu或者是单个gpu时，这时加载模型的方式有区别了。
```
#model_path是训练保存的模型文件
state_dict = torch.load(model_path)  
#create new OrderedDict that does not contain `module.
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # print("layer_name:",k)
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
#load params
model.load_state_dict(new_state_dict)
```
上述就是加载多个gpu训练的模型。使用gpu和cpu加载、训练模型的区别可以自己查询。<br>
## 后记
有问题随时交流哦。[chenyuan72530@gmail.com ]
