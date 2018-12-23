# deeplabv3_pytorch
deeplab v3 implement in pytorch

## 环境要求
pytorch==1.0.0<br>
torchvision=0.2.1<br>
torchsummaryX<br>
glob<br>
imageio<br>
其中torchsummaryX的功能和keras中model.summary()一样，可以打印出来model的结构，查看结构是否正确。安装方式如下：<br>
```
pip install torchsummaryX<br>
```
[使用方法参考](https://github.com/nmhkahn/torchsummaryX）<br>
另外介绍几个相关的可视化工具，torchsummary(类似torchsummaryX),[graphviz](https://github.com/szagoruyko/pytorchviz)<br>
## 用法
代码结构<br>
```
models.py
```
修改最后的部分：<br>
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
可以训练自己的模型（数据集加载部分已完成，具体格式待介绍）<br>
有问题随时交流哦。[chenyuan72530@gmail.com ]
