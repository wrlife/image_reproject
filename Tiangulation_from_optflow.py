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

        
        # if name.endswith('.pfm') or name.endswith('.PFM'):
        #     return readPFM(name)[0][:,:,0:2]

        f = open(name, 'rb')

        # header = f.read(4)
        # if header.decode("utf-8") != 'PIEH':
        #     raise Exception('Flow file header does not contain PIEH')

        # width = np.fromfile(f, np.int32, 1).squeeze()
        # height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, self.width * self.height * 2).reshape((self.height, self.width, 2))

        return flow.astype(np.float32)  
            
    def projective_inverse_warp(self,datafolder):
        
        import pdb;pdb.set_trace()
        self.colmapfile = datafolder+"/images.txt"
        I = pil.open(self.img)

        I = np.array(I)
        I1 = I[:,0:int(I.shape[1]/2),:]
        I2 = I[:,int(I.shape[1]/2):,:]
        #I2 = pil.open(self.img2)

        #I2 = np.array(I2)
        #I1 = np.array(I1)

        self.width = I1.shape[1]
        self.height = I1.shape[0]

        flow = self.readFlow(self.optflow)

        
        r1,t1,_,_=util.get_camera_pose(self.colmapfile,'frame'+self.img.split('/')[-1].split('.')[0].split('_')[0]+'.jpg')
        r2,t2,_,_=util.get_camera_pose(self.colmapfile,'frame'+self.img.split('/')[-1].split('.')[0].split('_')[1]+'.jpg')


        K = np.array([[float(567.239),0,float(376.04)],[0,float(261.757),float(100.961)],[0,0,1]])

        
        P1 = np.dot(K,np.append(inv(r1),np.dot(inv(r1),-t1.reshape(3,1)),1))
        P2 = np.dot(K,np.append(inv(r2),np.dot(inv(r2),-t2.reshape(3,1)),1))

        
        # depth = np.fromfile('/home/wrlife/project/deeplearning/SfMLearner/data/pairwise_good_maxnorm/case003/frame3330_3340.jpg_z.bin',dtype = np.float32).reshape(I1.shape[0],I1.shape[1])
        # x,y = util.get_camera_grid(I1.shape[1],I1.shape[0], float(376.04),float(100.961),float(567.239),float(261.757))

        # points = np.dstack((x,y,np.ones_like(x)))*depth.reshape(I1.shape[0],I1.shape[1],1)


        # points = points.reshape(I1.shape[0]*I1.shape[1],3)
        # points = np.transpose(points)
        
        # pad = np.ones([1,I1.shape[0]*I1.shape[1]])
        # points = np.append(points,pad,0)
        # points_tgt = np.dot(P1,points)

        # src_at_tgt = util.world2cam(points_tgt[0:3,:],float(376.04),float(100.961),float(567.239),float(261.757))
        
        
        x_coord1 = np.repeat(np.reshape(np.linspace(0, I1.shape[1]-1, I1.shape[1]),[1,I1.shape[1]]),I1.shape[0],0)
        y_coord1 = np.repeat(np.reshape(np.linspace(0, I1.shape[0]-1, I1.shape[0]),[I1.shape[0],1]),I1.shape[1],1)


        x_coord2 = x_coord1+flow[:,:,0]
        y_coord2 = y_coord1+flow[:,:,1]
        

        #import pdb;pdb.set_trace()

        #Triangulation
        for i in range(I1.shape[0]):
            for j in range(I1.shape[1]):
                #import pdb;pdb.set_trace()
                A = np.array((x_coord1[i,j]*P1[2,:]-P1[0,:],
                              y_coord1[i,j]*P1[2,:]-P1[1,:],
                              x_coord2[i,j]*P2[2,:]-P2[0,:],
                              y_coord2[i,j]*P2[2,:]-P2[1,:]))
                A[0,:] = A[0,:]/np.linalg.norm(A[0,:])
                A[1,:] = A[1,:]/np.linalg.norm(A[1,:])
                A[2,:] = A[2,:]/np.linalg.norm(A[2,:])
                A[3,:] = A[3,:]/np.linalg.norm(A[3,:])

                U, s, V = np.linalg.svd(A, full_matrices=True)

                if(i+j==0):
                    X = np.expand_dims(V[-1,0:3]/V[-1,-1],axis=0)
                    #X1 = np.expand_dims(V[0:3,-1],axis=0)
                else:
                    X = np.append(X,np.expand_dims(V[-1,0:3]/V[-1,-1],axis=0),axis=0)
                    #X1 = np.append(X1,np.expand_dims(V[0:3,-1],axis=0),axis=0)
                

        #ttt=np.reshape(X,(240,720,3))
        import pdb;pdb.set_trace()            
        util.save_sfs_ply('test.xyz', X)
        util.save_sfs_ply('test.xyz', X1)
        #import pdb;pdb.set_trace()
        # f, axarr = plt.subplots(3,1)

        # axarr[0].imshow(I1)
        # axarr[0].set_yticklabels([])
        # axarr[0].set_ylabel('Static image')

        # axarr[1].imshow(I_warp)
        # axarr[1].set_yticklabels([])
        # axarr[1].set_ylabel('warpped image')


        # axarr[2].imshow(I2)
        # axarr[2].set_yticklabels([])
        # axarr[2].set_ylabel('Moving image')
        
        # #plt.show()

        # plt.savefig(self.img1.split('/')[-1])





if __name__ == '__main__':

    # test = Reprojectimage('filenames.txt')
    # ops = test.readTupleList()
    # for ent in ops:
        
    #     test.img1 = ent[0]
    #     test.img2 = ent[1]
    #     test.optflow = ent[2]
    #     test.projective_inverse_warp('/home/wrlife/project/Unsupervised_Depth_Estimation/scripts/data/goodimages/case003/sfm_results/')
    test = Reprojectimage('filenames.txt')

    test.img = "/home/wrlife/Desktop/test/5941_5981.jpg"
    test.optflow = "/home/wrlife/Desktop/test/5941_5981.flo.flo"
    test.projective_inverse_warp('/home/wrlife/Desktop/test/')