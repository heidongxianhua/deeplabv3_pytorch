from deeplabv3_pytorch import deeplab_v3
import torch 
import os
from torchvision import transforms
from  DataLoader import TrainData,ToTensor,Normalize
from torch.autograd import Variable
import numpy as np 
import  torch.functional as F



from collections import OrderedDict
import matplotlib.pyplot as plt
import imageio
def test():
    input_shape=(720,1280) 
    backbone="xception"
    num_classes=3
    batch=1
    model = deeplab_v3(input_shape,backbone,num_classes)

    model.cuda(0)

    model_save_path = "/media/deeplearning/5e17034f-b03e-4b6f-8fbd-052df2b30327/chen/code/deeplab/check_points"
    model_name  ="Wed_Jan__2_01_02__xception.pth"
    model_path = os.path.join(model_save_path,model_name)
    # model.load_state_dict(torch.load(model_path))


    # original saved file with DataParallel
    state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # print("layer_name:",k)
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)



    test_path = "/home/deeplearning/code/BDD100K/dataset/bdd100k_images/images/100k"
    label_path = "/home/deeplearning/code/BDD100K/dataset/bdd100k_drivable_maps/drivable_maps/labels"
    datas = TrainData(test_path,label_path,True,transforms.Compose([ToTensor(),
                        Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]))
    dataset =torch.utils.data.DataLoader(datas,batch_size=batch,shuffle=True,num_workers=4)
    criterion = torch.nn.CrossEntropyLoss()  
    save_path = "./test_images"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    color_maps={0:(30,0,0),1:(0,30,0),2:(0,0,30)}
    with model.eval() and torch.no_grad():
        for i,data in enumerate(dataset,0):
            batch_sample = data
            batch_data = batch_sample["image"]

            batch_label = batch_sample["label"]
            batch_data,batch_label = Variable(batch_data.cuda()), Variable(batch_label.cuda()) #将数据从tensor转换为variable

            output = model(batch_data)
            # F.CrossEntropyLoss()
            loss = criterion(output,batch_label)
            pred = output.data.max(1,keepdim=True)[1]  #the location
            accuracy = pred.eq(batch_label.view_as(pred)).sum()
            pred = pred.data
            div_sum = (output.shape[0]*output.shape[-1]*output.shape[-2])
            accuracy = float(accuracy)/float(div_sum)
            # accuracy = accuracy*1.0/(output.shape[0]*output.shape[-1]*output.shape[-2])
            if output.shape[0]==1:
                image_name = batch_sample["name"][0]
                image_name = image_name.split(".")[0]+"_test.png"
                image_name = os.path.join(save_path,image_name)
                image = batch_data.data.squeeze(0)
                image = image.cpu().numpy()
                height,width = image.shape[-2],image.shape[-1]
                new_image = np.zeros((height,width,3))
                for i in range(height):
                    for j in range(width):
                        # print("type",type(image[:,i,j]))
                        new_image[i,j,:] =image[:,i,j]+np.array(color_maps[int(pred[0,0,i,j])])
                new_image = np.array(new_image,np.uint8)
                imageio.imwrite(image_name,new_image)
                # plt.figure("frame")
                # plt.imshow(new_image)
                # plt.show()
            print("loss:{},accuracy:{},div_sum:{}".format(loss,accuracy,div_sum))


test()

# x = torch.randn(4,10,30,30)
# y = F.Softmax(x)
# print(x)