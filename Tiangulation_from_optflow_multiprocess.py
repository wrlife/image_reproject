import util
import numpy as np
import PIL.Image as pil
from numpy.linalg import inv
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from multiprocessing import Process, Lock, Pool
from multiprocessing.sharedctypes import Array
from functools import partial

        

def readTupleList(filelist):
    filename = filelist
    list = []
    for line in open(filename).readlines():
        if line.strip() != '':
            list.append(line.split())

    return list

def readFlow(name,width,height):

    
    # if name.endswith('.pfm') or name.endswith('.PFM'):
    #     return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    # header = f.read(4)
    # if header.decode("utf-8") != 'PIEH':
    #     raise Exception('Flow file header does not contain PIEH')

    # width = np.fromfile(f, np.int32, 1).squeeze()
    # height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)  

def DLT_triangulate(x_coord1,y_coord1,x_coord2,y_coord2,P1,P2,X,h):
    #import pdb;pdb.set_trace()
    A = np.array((x_coord1[h]*P1[2,:]-P1[0,:],
                  y_coord1[h]*P1[2,:]-P1[1,:],
                  x_coord2[h]*P2[2,:]-P2[0,:],
                  y_coord2[h]*P2[2,:]-P2[1,:]))
    A[0,:] = A[0,:]/np.linalg.norm(A[0,:])
    A[1,:] = A[1,:]/np.linalg.norm(A[1,:])
    A[2,:] = A[2,:]/np.linalg.norm(A[2,:])
    A[3,:] = A[3,:]/np.linalg.norm(A[3,:])

    U, s, V = np.linalg.svd(A, full_matrices=True)

    X[int(h*3)]=V[-1,0]/V[-1,-1]
    X[int(h*3+1)]=V[-1,1]/V[-1,-1]
    X[int(h*3+2)]=V[-1,2]/V[-1,-1]



def f(self,x):
    return x*x        
        
def projective_inverse_warp(img1,img2,optflow,datafolder):
    
    import pdb;pdb.set_trace()
    colmapfile = datafolder+"/images.txt"
    I1 = pil.open(img1)

    # I = np.array(I)
    # I1 = I[:,0:int(I.shape[1]/2),:]
    # I2 = I[:,int(I.shape[1]/2):,:]
    I2 = pil.open(img2)

    I2 = np.array(I2)
    I1 = np.array(I1)

    width = I1.shape[1]
    height = I1.shape[0]

    flow = readFlow(optflow,width,height)

    
    r1,t1,_,_=util.get_camera_pose(colmapfile,img1.split('/')[-1])
    r2,t2,_,_=util.get_camera_pose(colmapfile,img2 .split('/')[-1])


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
    
    x_coord1 = np.reshape(x_coord1,-1)
    y_coord1 = np.reshape(y_coord1,-1)
    x_coord2 = np.reshape(x_coord2,-1)
    y_coord2 = np.reshape(y_coord2,-1)
    import pdb;pdb.set_trace()
    # lock = Lock()
    # X = Array('f',240*720*3,lock=lock)

    # pool = Pool(6)
    h = I1.shape[0]*I1.shape[1]
    # partialDLT = partial(DLT_triangulate, x_coord1=x_coord1,y_coord1=y_coord1,x_coord2=x_coord2,y_coord2=y_coord2,P1=P1,P2=P2,X=X,h=h)
    
    # mandelImg = pool.map(partialDLT, xrange(h))

    X = np.memmap('test', dtype=np.float32,
                     shape=h*3, mode='w+')  

    Parallel(n_jobs=8)(delayed(DLT_triangulate)(x_coord1,y_coord1,x_coord2,y_coord2,P1,P2,X,i)
                   for i in range(h))

    X_cam = np.array(X)
    X_cam=np.reshape(X_cam,(h,3))
    X_cam = inv(r1)*X-np.dot(inv(r1),-t1.reshape(3,1))
    X_cam=np.reshape(X_cam,(240,720,3))
    depth = np.maximum(X_cam[:,:,2], 1e-6)


    #m_pool = Pool(4)
    #Triangulation
    # for i in range(I1.shape[0]):
    #     #import pdb;pdb.set_trace()
    #     for j in range(I1.shape[1]):

    #         #import pdb;pdb.set_trace()
    #         # self.i = i
    #         p = Process(target=DLT_triangulate, args=(x_coord1[i,j],y_coord1[i,j],x_coord2[i,j],y_coord2[i,j],P1,P2,X,i,j))
            
    #         p.start()
    #         p.join()
        # print(p.map(f, range(I1.shape[1])))
        # m_pool.map(DLT_triangulate,range(I1.shape[1]))
        #Parallel(n_jobs=4)(delayed(self.DLT_triangulate)() for j in range(I1.shape[1]))
        #Parallel(n_jobs=4)(delayed(unwrap_self)(j) for j in zip([self]*I1.shape[1],range(I1.shape[1])))
            #tmp_point = self.DLT_triangulate(x_coord1[i,j],y_coord1[i,j],x_coord2[i,j],y_coord2[i,j],P1,P2)

            #X[i,j]=tmp_point

            # if(i+j==0):
            #     X = np.expand_dims(V[-1,0:3]/V[-1,-1],axis=0)
            #     #X1 = np.expand_dims(V[0:3,-1],axis=0)
            # else:
            #     X = np.append(X,np.expand_dims(V[-1,0:3]/V[-1,-1],axis=0),axis=0)                

    

    #Get depth map
    # X_cam = inv(r1)*X-np.dot(inv(r1),-t1.reshape(3,1))
    # X_cam=np.reshape(X_cam,(240,720,3))

    # depth = np.maximum(X_cam[:,:,2], 1e-6)




    import pdb;pdb.set_trace()            
    #util.save_sfs_ply(self.img1.split('/')[-1]+'.ply', ttt)

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


    ops = readTupleList('filenames.txt')
    for ent in ops:
        
        img1 = ent[0]
        img2 = ent[1]
        optflow = ent[2]
        projective_inverse_warp(img1,img2,optflow,'/home/wrlife/project/Unsupervised_Depth_Estimation/scripts/data/goodimages/case003/sfm_results/')
    # test = Reprojectimage('filenames.txt')

    # test.img = "/home/wrlife/Desktop/test/5941_5981.jpg"
    # test.optflow = "/home/wrlife/Desktop/test/5941_5981.flo.flo"
    # test.projective_inverse_warp('/home/wrlife/Desktop/test/')