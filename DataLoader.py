import torch.utils.data 
import os
import glob
import imageio
from torchvision import transforms, utils





###Normalize归一化:transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# (mean,std)
#mean = [0.485, 0.456, 0.406]
#        std = [0.229, 0.224, 0.225]
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
class ToTensor_back(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(landmarks)}

class ToTensor(object):
    def __init__(self):
        self.obj = transforms.ToTensor()
    def __call__(self,sample):
        image, landmarks = sample['image'], sample['label']
        # image = obj(image)
        landmarks = torch.from_numpy(landmarks)
        landmarks = landmarks.long()
        return {'image': self.obj(image),
                'label': landmarks}


class Normalize(object):
    def __init__(self,mean,std):
        # self.mean = mean
        # self.std = std
        self.obj = transforms.Normalize(mean,std)

    def __call__(self,sample):
        image, landmarks = sample['image'], sample['label']
        # image = obj(image)
        # landmarks = landmarks
        return {'image': self.obj(image),
                'label': landmarks}





import numpy as np
class TrainData(torch.utils.data.Dataset):
    def __init__(self,train_path,label_path,train=True,transform=None):
        super(TrainData,self).__init__()
        # self.image_path = os.path.join(self.data_path,"images")
        # self.label_path = os.path.join(self.data_path,"labels")
        self.images_path = os.path.join(train_path,"train")
        self.labels_path = os.path.join(label_path,"train")

        self.images_list = glob.glob(os.path.join(self.images_path,"*.jpg"))
        self.images_list = self.images_list
        self.labels_list = glob.glob(os.path.join(self.labels_path,"*.png"))
        self.nums = min(len(self.images_list),len(self.labels_list))
        self.transform =transform
        self.train = train
    def __getitem__(self,index):
        while True:
            self.image_name_ = os.path.basename(self.images_list[index])
            self.image_name = os.path.basename(self.images_list[index])
            self.temp_name = self.image_name.split(".")[0]+"_drivable_id"+".png"
            self.label_path = os.path.join(self.labels_path,self.temp_name)
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
        # label = np.expand_dims(label,0)
        sample = {"image":image,"label":label}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample["image"] = torch.from_numpy(sample["image"]).float()
            sample["label"] = torch.from_numpy(sample["label"]).float()
        sample["name"] =self.image_name_
        return sample
    def __len__(self):
        return self.nums

if __name__=="__main__":
    datas = TrainData("./data/",transforms.Compose([ToTensor(),
                        Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]))
    dataset = dtorch.utils.data.DataLoader(datas,16,True,4)