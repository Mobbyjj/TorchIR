# to load the data from ACDC dataset
from torch.utils.data import Dataset
import os
from os import listdir
import nibabel as nib
import numpy as np

# adopted from MotionNet
class ACDCTrainDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(ACDCTrainDataset, self).__init__()
        self.data_path = data_path
        self.subject_list = [f for f in sorted(listdir(self.data_path))]
        
        # data augmentation
        self.transform = transform
        
    def _crop(self,Img,sy,sx):
        img = Img.get_fdata()
        [ny,nx,nslice] = np.shape(img)
        img = img[round((ny-sy)/2):round((ny+sy)/2),round((nx-sx)/2):round((nx+sx)/2),0:nslice]
        return img

    def _norm_inten(self,img):
        # for 3D images, just normalized slice by slice
        # nx, ny, nslice
        self.img = img
        self.img = np.transpose(self.img,[2,0,1])
        [nslice, nx, ny] = np.shape(self.img)
        self.normedimg = np.zeros((nslice, nx, ny))
        for i in range(nslice):
            img2 = (self.img[i] - self.img[i].min())/(self.img[i].max() - self.img[i].min())
            self.normedimg[i] = img2
        self.normedimg = np.transpose(self.normedimg,[1,2,0])
        return self.normedimg

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        
        idx = index//7
        idxslice = idx % 7
        self.subj_id = self.subject_list[idx]
        self.subj_path = f'{self.data_path}/{self.subj_id}/'
        GTfiles = [f for f in os.listdir(self.subj_path) if f.endswith('_gt.nii.gz') ]
        GTfiles.sort()
        self.dsfilename = GTfiles[0][:-10] + '.nii.gz'
        self.esfilename = GTfiles[1][:-10] + '.nii.gz'
        self.dsfile = nib.load(self.subj_path + self.dsfilename)
        self.esfile = nib.load(self.subj_path + self.esfilename)
        self.esseg = nib.load(self.subj_path + GTfiles[1])
        
        m1 = self._crop(self.dsfile, 128, 128)
        f1 = self._crop(self.esfile, 128, 128)
        # f1seg = self._crop(self.esseg,128, 128)
        
        m1norm = self._norm_inten(m1)
        f1norm = self._norm_inten(f1)
        
        x, y = m1norm[np.newaxis, ..., idxslice], f1norm[np.newaxis,..., idxslice]
        # yseg = f1seg[np.newaxis,np.newaxis, ..., idxslice]
    
        #image_bank = np.concatenate((x, y), axis=1)
        #input = np.array(image_bank, dtype='float32')
        # target = np.array(yseg, dtype='int16')

        if self.transform:
            input, target = self.transform(input, target)

        #image = input[0,:1]
        #image_pred = input[0,1:]

        #return image, image_pred, target[0,0]
        return {"fixed":y.astype(np.float32), "moving":x.astype(np.float32)}
        

    def __len__(self):
        self.subject_list = sorted(os.listdir(self.data_path))
        return len(self.subject_list*7)
    
class ACDCValDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(ACDCValDataset, self).__init__()
        self.data_path = data_path
        self.subject_list = [f for f in sorted(listdir(self.data_path))]

        # data augmentation
        self.transform = transform
    
    def _crop(self,Img,sy,sx):
        img = Img.get_fdata()
        [ny,nx,nslice] = np.shape(img)
        img = img[round((ny-sy)/2):round((ny+sy)/2),round((nx-sx)/2):round((nx+sx)/2),0:nslice]
        return img
    
    def _norm_inten(self,img):
        # for 3D images, just normalized slice by slice
        # nx, ny, nslice
        self.img = img
        self.img = np.transpose(self.img,[2,0,1])
        [nslice, nx, ny] = np.shape(self.img)
        self.normedimg = np.zeros((nslice, nx, ny))
        for i in range(nslice):
            img2 = (self.img[i] - self.img[i].min())/(self.img[i].max() - self.img[i].min())
            self.normedimg[i] = img2
        self.normedimg = np.transpose(self.normedimg,[1,2,0])
        return self.normedimg
    

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        #np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        idx = index//7
        idxslice = idx % 7
        self.subj_id = self.subject_list[idx]
        self.subj_path = f'{self.data_path}/{self.subj_id}/'
        GTfiles = [f for f in os.listdir(self.subj_path) if f.endswith('_gt.nii.gz') ]
        GTfiles.sort()
        self.esfilename = GTfiles[1][:-10] + '.nii.gz'
        self.esfile = nib.load(self.subj_path + self.esfilename)
        #self.esseg = nib.load(self.subj_path + GTfiles[1])
        f1 = self._crop(self.esfile, 128, 128)
        f1norm = self._norm_inten(f1)
        #target = self._crop(self.esseg, 128, 128)
        
        self.dsfilename = GTfiles[0][:-10] + '.nii.gz'
        self.dsfile = nib.load(self.subj_path + self.dsfilename)
        m1 = self._crop(self.dsfile, 128, 128)
        m1norm = self._norm_inten(m1)
        
        x, y = m1norm[np.newaxis, ..., idxslice], f1norm[np.newaxis, ..., idxslice]

        if self.transform:
            input, target = self.transform(input, target)

        return {"fixed":y.astype(np.float32), "moving":x.astype(np.float32)}

    def __len__(self):
        self.subject_list = sorted(os.listdir(self.data_path))
        return len(self.subject_list)