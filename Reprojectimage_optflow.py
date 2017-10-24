import util
import numpy as np
import PIL.Image as pil
from numpy.linalg import inv
import matplotlib.pyplot as plt

class Reprojectimage():

    def __init__(self, filelist):

        self.filelist = filelist;

    def readTupleList(self):
        filename = self.filelist
        list = []
        for line in open(filename).readlines():
            if line.strip() != '':
                list.append(line.split())

        return list

    def readFlow(self,name):

        
        if name.endswith('.pfm') or name.endswith('.PFM'):
            return readPFM(name)[0][:,:,0:2]

        f = open(name, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

        return flow.astype(np.float32)  
            
    def projective_inverse_warp(self):
        
        #import pdb;pdb.set_trace()
        I1 = pil.open(self.img1)

        I2 = pil.open(self.img2)

        I2 = np.array(I2)
        I1 = np.array(I1)
        flow = self.readFlow(self.optflow)



        
        x_coord = np.repeat(np.reshape(np.linspace(0, I1.shape[1]-1, I1.shape[1]),[1,I1.shape[1]]),I1.shape[0],0)
        y_coord = np.repeat(np.reshape(np.linspace(0, I1.shape[0]-1, I1.shape[0]),[I1.shape[0],1]),I1.shape[1],1)


        x_coord = x_coord+flow[:,:,0]
        y_coord = y_coord+flow[:,:,1]
        

        #Bilinear interpolation
        I_warp = self.bilinear_interpolate(I2,np.reshape(x_coord,-1), np.reshape(y_coord,-1))

        I_warp = I_warp.reshape(I1.shape[0],I1.shape[1],3)
        I_warp = I_warp.astype(np.uint8)

        I_new = 0.2*I1+0.8*I_warp
        I_new = I_new.astype(np.uint8)
        
        f, axarr = plt.subplots(3,1)

        axarr[0].imshow(I1)
        axarr[0].set_yticklabels([])
        axarr[0].set_ylabel('Static image')

        axarr[1].imshow(I_warp)
        axarr[1].set_yticklabels([])
        axarr[1].set_ylabel('warpped image')


        axarr[2].imshow(I2)
        axarr[2].set_yticklabels([])
        axarr[2].set_ylabel('Moving image')
        
        #plt.show()

        plt.savefig(self.img1.split('/')[-1])





if __name__ == '__main__':

    test = Reprojectimage('filenames.txt')
    ops = test.readTupleList()
    for ent in ops:
        test.img1 = ent[0]
        test.img2 = ent[1]
        test.optflow = ent[2]
        test.projective_inverse_warp()