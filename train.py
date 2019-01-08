from deeplabv3_pytorch import deeplab_v3
import torch 
import os
from torchvision import transforms
from  DataLoader import TrainData,ToTensor,Normalize
import torch.utils.data
import time
from tqdm import tqdm
def train(epoches=100):
    input_shape=(720,1280) 
    backbone="xception"
    num_classes=3
    batch=3
    model = deeplab_v3(input_shape,backbone,num_classes)
    train_path = "/home/deeplearning/code/BDD100K/dataset/bdd100k_images/images/100k"
    label_path = "/home/deeplearning/code/BDD100K/dataset/bdd100k_drivable_maps/drivable_maps/labels"
    datas = TrainData(train_path,label_path,True,transforms.Compose([ToTensor(),
                        Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]))

    dataset =torch.utils.data.DataLoader(datas,batch_size=batch,shuffle=True,num_workers=4)
    steps = len(dataset)//3
    print("steps",len(dataset))
    print("gpu",torch.cuda.device_count())

    # for i,data in enumerate(dataset,0):
    #         batch_sample = data
    #         batch_data = batch_sample["image"]
    #         batch_label = batch_sample["label"]
    #         print(batch_data.shape)
    #         print(batch_label.shape)
    #         break
    from torch.autograd import Variable
    from torch import optim
    # from torch import nn

    model = torch.nn.DataParallel(model, device_ids=[0,1,2], output_device=None, dim=0)
    model.cuda()
    
    criterion = torch.nn.CrossEntropyLoss()    
    # criterion = my_loss()
    model_save_path ="/media/deeplearning/5e17034f-b03e-4b6f-8fbd-052df2b30327/chen/code/deeplab/check_points"
    if True:
        model.load_state_dict(torch.load(os.path.join(model_save_path,"Thu_Jan__3_09_10__xception.pth")))
        # model.load(os.path.join(model_save_path,"Sat_Dec_29_22_xception.pth"))
        
    optimizer = optim.Adam(model.parameters(), 1e-4,
                            weight_decay=5e-4)
    name = time.asctime( time.localtime(time.time()) )
    name = name.replace(" ","_").replace(":","_")[:-7]
    from tensorboardX import SummaryWriter
    with open(os.path.join(model_save_path,name+"check"+".txt"),"w") as f , SummaryWriter(log_dir="run_1") as writer:
        for epoch in tqdm(range(epoches)):
            running_loss = 0
            for i,data in enumerate(dataset,0):
                batch_sample = data
                batch_data = batch_sample["image"]
                batch_label = batch_sample["label"]
                batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
                inputs = Variable(batch_data)
                label = Variable(batch_label)  #将 数据从tensor转换为variable
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output,label)
                loss.backward()
                optimizer.step()
                writer.add_scalar("loss/step",loss,i+steps*epoch)
                if i %100 ==0:
                    print("******epoch:{}/{},step:{}/{},loss:{}".format(epoch,epoches,i,steps,loss))
                if epoch>0 and epoch %2 ==0 and i==0:
                    print("******epoch:{}/{},saving model.....".format(epoch,epoches))
                    name = time.asctime( time.localtime(time.time()) )
                    name = name.replace(" ","_").replace(":","_")[:-7]
                    temp_path = os.path.join(model_save_path,name+"_xception.pth")
                    torch.save(model.state_dict(), temp_path)
                    f.write("epoch: "+str(epoch)+"  loss: "+str(loss.data)+"  file_name: "+name +"\n")
                print("epoch:{},step:{},loss:{}".format(epoch,i,loss.data))

    # print("-------------")

train(epoches=500)



#model save 
# methos 1:
        #only save the parameters
        # #保存
        # torch.save(the_model.state_dict(), PATH)
        # #读取
        # the_model = TheModelClass(*args, **kwargs)
        # the_model.load_state_dict(torch.load(PATH))
# methos 2:
        # #保存,save  the whole model
        # torch.save(the_model, PATH)
        # #读取
        # the_model = torch.load(PATH)
# import time 
# name = time.asctime( time.localtime(time.time()) )
# name = name.replace(" ","_").replace(":","_")[:-7]
# print(name)
# 格式化成2016-03-20 11:45:39形式
# Sat_Dec_29_21_58_47_2018   Sat_Dec_29_21_  Sat_Dec_29_21
