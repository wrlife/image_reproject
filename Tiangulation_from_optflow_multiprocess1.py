import util
import numpy as np
import PIL.Image as pil
from numpy.linalg import inv
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from multiprocessing import Process, Lock, Pool
from multiprocessing.sharedctypes import Array
from functools import partial
import os, sys
        

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
        
def projective_inverse_warp(img,optflow,datafolder):
    

    colmapfile = datafolder+"images.txt"
    I = pil.open(img)

    I = np.array(I)
    I1 = I[:,0:int(I.shape[1]/2),:]
    I2 = I[:,int(I.shape[1]/2):,:]

    width = I1.shape[1]
    height = I1.shape[0]

    flow = readFlow(optflow,width,height)
    
    r1,t1,_,_=util.get_camera_pose(colmapfile,'frame'+img.split('/')[-1].split('.')[0].split('_')[0]+'.jpg')
    r2,t2,_,_=util.get_camera_pose(colmapfile,'frame'+img.split('/')[-1].split('.')[0].split('_')[1]+'.jpg')


    K = np.array([[float(567.239),0,float(376.04)],[0,float(261.757),float(100.961)],[0,0,1]])

    
    P1 = np.dot(K,np.append(inv(r1),np.dot(inv(r1),-t1.reshape(3,1)),1))
    P2 = np.dot(K,np.append(inv(r2),np.dot(inv(r2),-t2.reshape(3,1)),1))

    
    x_coord1 = np.repeat(np.reshape(np.linspace(0, I1.shape[1]-1, I1.shape[1]),[1,I1.shape[1]]),I1.shape[0],0)
    y_coord1 = np.repeat(np.reshape(np.linspace(0, I1.shape[0]-1, I1.shape[0]),[I1.shape[0],1]),I1.shape[1],1)


    x_coord2 = x_coord1+flow[:,:,0]
    y_coord2 = y_coord1+flow[:,:,1]
    
    x_coord1 = np.reshape(x_coord1,-1)
    y_coord1 = np.reshape(y_coord1,-1)
    x_coord2 = np.reshape(x_coord2,-1)
    y_coord2 = np.reshape(y_coord2,-1)


    I_warp,vmask = util.bilinear_interpolate(I,x_coord2, y_coord2)
    # I_warp = I_warp.reshape(I1.shape[0],I1.shape[1],3)
    # I_warp = I_warp.astype(np.uint8)
    # plt.imshow(I_warp)
    # plt.show()

    vmask = vmask.reshape(I1.shape[0],I1.shape[1])
    #vmask = np.repeat(vmask,3,axis=2)


    
    h = I1.shape[0]*I1.shape[1]


    X = np.memmap('test', dtype=np.float32,
                     shape=h*3, mode='w+')  

    Parallel(n_jobs=8)(delayed(DLT_triangulate)(x_coord1,y_coord1,x_coord2,y_coord2,P1,P2,X,i)
                   for i in range(h))


    X_cam = np.array(X)
    #X_cam=np.reshape(X_cam,(h,3))
    #X_cam = np.transpose(np.dot(inv(r1),np.transpose(X_cam))-np.dot(inv(r1),-t1.reshape(3,1)))
    X_cam=np.reshape(X_cam,(240,720,3))
    
    depth = np.maximum(X_cam[:,:,2], 1e-6)

    #depth = depth*vmask
    #X_cam[:,:,2] = depth
    util.save_sfs_ply(img+'.ply', X_cam)
    import pdb;pdb.set_trace()

    return np.expand_dims(depth,axis=2)





                





if __name__ == '__main__':


    # test.img = "/home/wrlife/Desktop/test/5941_5981.jpg"
    # test.optflow = "/home/wrlife/Desktop/test/5941_5981.flo.flo"
    # test.projective_inverse_warp('/home/wrlife/Desktop/test/')


    inputbase = '/home/wrlife/Desktop/test/'

    #ops = readTupleList('filenames.txt')
    lists = sorted(os.listdir(inputbase),reverse=True)



    for i in range(0,len(lists),2):
        import pdb;pdb.set_trace()

        optflow = inputbase+lists[i+1]

        img = inputbase+lists[i]

        
        depth = projective_inverse_warp(img,optflow,'./')

        if i==0:
            final_depth = depth
        else:
            final_depth = np.concatenate([final_depth,depth],axis=2)
    # test = Reprojectimage('filenames.txt')

    # test.img = "/home/wrlife/Desktop/test/5941_5981.jpg"
    # test.optflow = "/home/wrlife/Desktop/test/5941_5981.flo.flo"
    # test.projective_inverse_warp('/home/wrlife/Desktop/test/')