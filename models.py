#coding=utf-8
from torch import nn
import torch.nn.functional as F 
import torch
import math
#conv2d的参数(self, in_channels, out_channels, kernel_size, stride=1,
 #                padding=0, dilation=1, groups=1, bias=True):
class SepConv_BN(nn.Module):
    def __init__(self,input_channel,filters,stride=1,depth_ac= False,
                    rate=1,epsilon=1e-3,kernel_size=3,):
        super(SepConv_BN,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.depth_ac = depth_ac
       
        pad = (kernel_size + (kernel_size - 1) * (rate - 1))//2
        
        self.depth_conv = nn.Conv2d(input_channel,input_channel,kernel_size,
                                    stride=stride,padding=pad,
                                    dilation=rate,groups=input_channel,bias=False)
        self.bn1=nn.BatchNorm2d(input_channel)
        self.conv = nn.Conv2d(input_channel,filters,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(filters)
    
    def forward(self,x):
        if not self.depth_ac:
            x = self.relu(x)
        x = self.depth_conv(x)
        x = self.bn1(x)
        if self.depth_ac:
            x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)
        if self.depth_ac:
            x = self.relu(x)
        return x

class xception_block(nn.Module):
    def __init__(self,input_channel,depth_list,stride,skip_type,rate=1,
                    depth_activation=False,return_skip = False):
        super(xception_block,self).__init__()
        self.skip_type = skip_type
        self.return_skip = return_skip
        self.layer1 = SepConv_BN(input_channel,depth_list[0],1,depth_activation,rate)
        self.layer2 = SepConv_BN(depth_list[0],depth_list[1],1,depth_activation,rate)
        self.layer3 = SepConv_BN(depth_list[1],depth_list[2],stride,depth_activation,rate)
        if self.skip_type =="conv":
            self.short_cut = nn.Sequential(*[nn.Conv2d(input_channel,depth_list[-1],1,stride),
                                nn.BatchNorm2d(depth_list[-1])])
    
    def forward(self,x):
        # residual = x
        y = self.layer1(x)
        y = self.layer2(y)
        skip = y
        y = self.layer3(y)
        if self.skip_type =='conv':
            z = self.short_cut(x)
            # z = z+y
            z = torch.add(z,y)
        elif self.skip_type=="sum":
            # z = y+x
            z = torch.add(y,x)
        elif self.skip_type=="none":
            z = y 
        if self.return_skip:
            return z,skip
        return z

class BilinearUpsampling(nn.Module):
    def __init__(self,size=None,factor=None):
        if (not size) and (not factor):
            raise ValueError("size or factor is error!")
        self.size = size
        self.factor =factor
    def forward(self,x):
        if self.size is not None:
            return nn.UpsamplingBilinear2d(size=self.size)
        else:
            return nn.UpsamplingBilinear2d(scale_factor=self.factor)

class inverted_res_block(nn.Module):
    def __init__(self,input_channel,out_channel,stride,expansion,\
                skip_connection,block_id=1,rate=1):
        super(inverted_res_block,self).__init__()
        # self.in_channel = input_channel
        # self.out_channel = out_channel
        # self.expansion =expansion
        self.in_channel = input_channel
        self.skip = skip_connection
        self.block_id = block_id
        if self.block_id:
            self.pre = nn.Sequential(
                            nn.Conv2d(input_channel,input_channel*expansion,1,bias=False),
                            nn.BatchNorm2d(input_channel*expansion),
                            nn.ReLU6(inplace=True))
            self.in_channel = self.in_channel*expansion
        self.padding =  (3 + (3 - 1) * (rate - 1))//2
        self.b1 = nn.Sequential(
                    nn.Conv2d(self.in_channel,self.in_channel,3,stride,self.padding,rate,self.in_channel,False),
                    nn.BatchNorm2d(self.in_channel,),
                    nn.ReLU6(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(self.in_channel,out_channel,1,bias=False),
                    nn.BatchNorm2d(out_channel),)
    def forward(self,inputs):
        x = inputs
        if self.block_id:
            x = self.pre(x)
        x = self.b1(x)
        x = self.b2(x)
        if self.skip:
            x = torch.add(inputs,x)
        return x
 class my_loss(nn.Module):
    def __init__(self):
        super(my_loss,self).__init__()
    
    def forward(self,y_true,y_pred):
        batch_size = y_pred.size[0]
        loss = (y_true-y_pred)**2/batch_size
        return loss

class my_loss(nn.Module):
    def __init__(self):
        super(my_loss,self).__init__()
    
    def forward(self,y_true,y_pred):
        batch_size = y_pred.size[0]
        loss = (y_true-y_pred)**2/batch_size
        return loss
   
class deeplab_v3(nn.Module):
    def __init__(self,input_shape=(512,512),backbone="xception",num_classes=10,OS=16):
        super(deeplab_v3,self).__init__()
        if backbone not in {"xception","mobilenetv2"}:
            raise ValueError("The `backbone` argument should be `xception` or \
                                `mobilenetv2`")
        if OS not in {8,16}:
            raise ValueError("The `OS` argument should be  8 or 16 ")
        self.backbone = backbone
        input_shape =input_shape
        self.in_channel = 0
        if self.backbone=="xception":
            if OS == 8:
                entry_block3_stride = 1
                middle_block_rate = 2  # ! Not mentioned in paper, but required
                exit_block_rates = (2, 4)
                atrous_rates = (12, 24, 36)
            else:
                entry_block3_stride = 2
                middle_block_rate = 1
                exit_block_rates = (1, 2)
                atrous_rates = (6, 12, 18) 

            self.pre_block = nn.Sequential(
                        nn.Conv2d(3,32,3,2,padding=(3-1)//2,bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32,64,3,1,padding=(3-1)//2,bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True))
            self.block1 = xception_block(64,[128,128,128],2,"conv")
            self.block2 = xception_block(128,[256,256,256],2,"conv",return_skip=True)
            self.block3 = xception_block(256,[728,728,728],entry_block3_stride,"conv")
            self.block_16 =[]
            for i in range(16):
                self.block_16.append(xception_block(728,[728,728,728],1,"sum",rate=middle_block_rate))
            self.block4 = nn.Sequential(*self.block_16)
            self.block5 = xception_block(728,[728,1024,1024],1,"conv",rate=exit_block_rates[0],)
            self.block6 = xception_block(1024,[1536,1536,2048],1,"none",rate=exit_block_rates[1])
            self.in_channel = 2048
        else:   #backbone is mobilenetv2
            OS = 8
            self.pre_block = nn.Sequential(
                                    nn.Conv2d(3,32,3,2,padding=(3-1)//2,bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU6(inplace=True))
            self.block0 = inverted_res_block(32,16,1,1,False,0)
            self.block1 = inverted_res_block(16,24,2,6,False,1) 
            self.block2 = inverted_res_block(24,24,1,6,True,2) 
            self.block3 = inverted_res_block(24,32,2,6,False,3) 
            self.block4 = inverted_res_block(32,32,1,6,True,4) 
            self.block5 = inverted_res_block(32,32,1,6,True,5) 
            self.block6 = inverted_res_block(32,64,1,6,False,6) 
            self.block7 = inverted_res_block(64,64,1,6,True,7,2) 
            self.block8 = inverted_res_block(64,64,1,6,True,8,2)
            self.block9 = inverted_res_block(64,64,1,6,True,9,2)
            self.block10 = inverted_res_block(64,96,1,6,False,10,2)
            self.block11 = inverted_res_block(96,96,1,6,True,11,2) 
            self.block12 = inverted_res_block(96,96,1,6,True,12,2) 
            self.block13 = inverted_res_block(96,160,1,6,False,13,2)
            self.block14 = inverted_res_block(160,160,1,6,True,14,4)
            self.block15 = inverted_res_block(160,160,1,6,True,15,4)
            self.block16 = inverted_res_block(160,320,1,6,False,16,4)
            blocks = []
            for i in range(17):
                name = eval("self.block"+str(i))
                blocks.append(name)
            self.blocks = nn.Sequential(*blocks)
            self.in_channel = 320
        self.pos1 = nn.Sequential(*[
                    nn.AvgPool2d((input_shape[0]//OS,input_shape[1]//OS)),
                    nn.Conv2d(self.in_channel,256,1,bias=False),
                    # nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                   nn.UpsamplingBilinear2d(size=(input_shape[0]//OS,input_shape[1]//OS)), ])
        self.in_channel = 320
        self.s = nn.Sequential(*[
                        nn.Conv2d(self.in_channel,256,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)])
        self.in_channel = 256
        if self.backbone=="xception":
            self.p1 = SepConv_BN(2048,256,depth_ac=True,rate=atrous_rates[0],epsilon=1e-5)
            self.p2 = SepConv_BN(2048,256,depth_ac=True,rate=atrous_rates[1],epsilon=1e-5)
            self.p3 = SepConv_BN(2048,256,depth_ac=True,rate=atrous_rates[2],epsilon=1e-5)
            pos2_in = 256*5
        else:
            pos2_in = 256*2
        self.pos2 = nn.Sequential(*[
                        nn.Conv2d(pos2_in,256,1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5)])
        in_channel = 256
        if self.backbone=="xception": 
            self.up1 = nn.UpsamplingBilinear2d(size = (input_shape[0]//4,input_shape[0]//4))
            self.skip_conv1 = nn.Conv2d(256,48,1,bias=False)
            self.skip_bn1 = nn.BatchNorm2d(48,eps=1e-5)
            self.skip_relu = nn.ReLU(inplace=True)
            self.up_sepconv1 = SepConv_BN(in_channel+48,256,depth_ac=True,epsilon=1e-5)
            self.up_sepconv2 = SepConv_BN(256,256,depth_ac=True,epsilon=1e-5)
        in_channel = 256
        self.up1_conv2 = nn.Conv2d(in_channel,num_classes,1)
        self.up2 = nn.UpsamplingBilinear2d(size= (input_shape[0],input_shape[1]))

        initialize the parameters,固定写法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        if self.backbone == "xception":
            x = self.pre_block(x)
            x = self.block1(x)
            x,skip = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            c1 = x = self.block6(x)
        else:
            x = self.pre_block(x)
            x = self.blocks(x)
        y1 = self.pos1(x)     #b4
        y2 = self.s(x)       #b0
        if self.backbone == "xception":
            p1 = self.p1(x)   #b1
            p2 = self.p2(x)   #b2
            p3 = self.p3(x)    #b3
            c2 = x =torch.cat([y1,y2,p1,p2,p3],1)   #c2
        else:
            x = torch.cat([y1,y2],1)
        x = self.pos2(x)
        if self.backbone == "xception":
            x = self.up1(x)
            dec_skip1 = self.skip_conv1(skip)
            dec_skip1 = self.skip_bn1(dec_skip1)
            dec_skip1 = self.skip_relu(dec_skip1)
            x = torch.cat([x,dec_skip1],1)
            x= self.up_sepconv1(x)
            x =self.up_sepconv2(x)
            
        x = self.up1_conv2(x)
        x = self.up2(x)
        return x

    def trainer(self,dataset,epoches=5000):
        from torch.autograd import Variable
        from torch import optim 
        model = deeplab_v3()

        criterion = nn.CrossEntropyLoss()    
        # criterion = my_loss()
        optimizer = optim.SGD(model.parameters(), 1e-3,
                                momentum=0.9,
                                weight_decay=5e-4)
        for epoch in range(epoches):
            running_loss = 0
            for i,data in enumerate(dataset,0):
                batch_sample = data
                batch_data = batch_sample["image"]
                batch_label = batch_sample["label"]
                inputs = Variable(batch_data)
                label = Variable(batch_label)  #将数据从tensor转换为variable
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output,label)
                loss.backward()
                optimizer.step()
                if i%100 ==0:
                    print("epoch/epoches:{}/{},step:{},loss:{}".format(epoch,epoches,i,loss))
            # if epoch%100 ==0:
            print("*****epoch/epoches:{}/{},loss:{}".format(epoch,epoches,loss))
#对于检测的任务，上述不同的地方在 criterion的选择不同，可能选择自定义损失函数，与自定义层的写法一致 
#然后将上面的nn.CrossEntropyLoss() 换成 my_loss()即可

import torch.utils.data as data
import os
import glob
import imageio
from torchvision import transforms, utils
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transforms.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'label': landmarks}
#数据增强：转换为tensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(landmarks)}

class TrainData(data.Dataset):
    def __init__(self,data_path,transform=None):
        super(TrainData,self).__init__()
        self.data_path = data_path
        self.image_path = os.path.join(self.data_path,"images")
        self.label_path = os.path.join(self.data_path,"labels")
        self.images_list = glob.glob(os.path.join(self.image_path,"*.png"))
        self.labels_list = glob.glob(os.path.join(self.label_path,"*.png"))
        self.nums = min(len(self.images_list,self.labels_list))
        self.transform =transform
    def __getitem__(self,index):
        while True:
            self.image_name = os.path.basename(self.images_list[index])
            self.label_path = os.path.join(self.data_path,"labels",self.image_name)
            if self.label_path in self.labels_list:
                break
            else:
                if index+1 < self.nums:
                    index +=1
                else:
                    index -=1
        self.image_name = self.images_list[index]
        image = imageio.imread(self.image_name)
        # image = torch.from_numpy(image).float() #需要转成float
        label = imageio.imread(self.label_path)
        sample = {"image":image,"label":label}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample["image"] = torch.from_numpy(sample["image"]).float()
            sample["label"] = torch.from_numpy(sample["label"]).float()
        return sample
    def __len__(self):
        return self.nums

def _main():
    model_name = "xception"
    input_shape = (512,512) # 默认值，可修改,只是表示图像的H*W，通道数为3
    num_class = 10 #
    deep_model = deeplab_v3(input_shape,model_name,num_class)
    datas = TrainData("./data/",transforms.Compose([Rescale(512),ToTensor()]))
    dataset = data.DataLoader(datas,16,True,4)
    deep_model.trainer(dataset,)


def model_framework():
    from torchsummaryX import summary
    model = deeplab_v3()
    summary(model,torch.zeros(1,3,512,512))     

if __name__=="__main__":
    # model_framework()
    _main()
