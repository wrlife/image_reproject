import util
import numpy as np
import PIL.Image as pil
from numpy.linalg import inv
import matplotlib.pyplot as plt

class Reprojectimage():

	def __init__(self, datafolder):
		self.datafolder = datafolder
		self.colmapfile = datafolder+"/images.txt"


	def projective_inverse_warp(self):
        
		#import pdb;pdb.set_trace()
		I = pil.open(self.datafolder+'/frame1819.jpg')

		I_ori = pil.open(self.datafolder+'/frame1829.jpg')

		I_ori = np.array(I_ori)
		I = np.array(I)
		depth = np.fromfile(self.datafolder+'/frame1829.jpg_z.bin',dtype = np.float32).reshape(I.shape[0],I.shape[1])

		x,y = self.get_camera_grid(I.shape[1],I.shape[0], float(376.04),float(100.961),float(567.239),float(261.757))

		points = np.dstack((x,y,np.ones_like(x)))*depth.reshape(I.shape[0],I.shape[1],1)

		#util.save_sfs_ply('test.ply',points)

		points = points.reshape(I.shape[0]*I.shape[1],3)
		points = np.transpose(points)
		pad = np.ones([1,I.shape[0]*I.shape[1]])
		points = np.append(points,pad,0)

		#Get camera translation matrix
		r1,t1,_,_=util.get_camera_pose(self.colmapfile,'frame1819.jpg')
		r2,t2,_,_=util.get_camera_pose(self.colmapfile,'frame1829.jpg')

		pad = np.array([[0, 0, 0, 1]])

		homo1 = np.append(r1,t1.reshape(3,1),1)
		homo1 = np.append(homo1,pad,0)

		homo2 = np.append(r2,t2.reshape(3,1),1)
		homo2 = np.append(homo2,pad,0)

		src2tgt_proj = inv(homo1)*homo2

		#Translate points to tgt cam coor
		points_tgt = np.dot(src2tgt_proj,points)

		#util.save_sfs_ply('test1.ply',points_tgt[0:3,:].reshape(I.shape[0],I.shape[1],3))

		#Translate points to tgt cam plane
		src_at_tgt = self.world2cam(points_tgt[0:3,:],float(376.04),float(100.961),float(567.239),float(261.757))

		import pdb;pdb.set_trace()
		#Bilinear interpolation
		I_warp = self.bilinear_interpolate(I,src_at_tgt[0,:], src_at_tgt[1,:])

		I_warp = I_warp.reshape(I.shape[0],I.shape[1],3)
		I_warp = I_warp.astype(np.uint8)

		I_new = 0.2*I_ori+0.8*I_warp
		I_new = I_new.astype(np.uint8)
		
		plt.imshow(I_new)

	def get_camera_grid(self,width,height,cx,cy,fx,fy):
	    return np.meshgrid(
	    	(np.arange(width)-cx)/fx,
	        (np.arange(height)-cy)/fy)
    

	def world2cam(self,points,cx,cy,fx,fy):
		camcorr = points[0:2,:]/points[2,:]
		camcorr[0,:]=camcorr[0,:]*fx+cx
		camcorr[1,:]=camcorr[1,:]*fy+cy
		return camcorr

	def bilinear_interpolate(self, im, x, y):
	    x = np.asarray(x)
	    y = np.asarray(y)

	    x0 = np.floor(x).astype(int)
	    x1 = x0 + 1
	    y0 = np.floor(y).astype(int)
	    y1 = y0 + 1

	    x0 = np.clip(x0, 0, im.shape[1]-1);
	    x1 = np.clip(x1, 0, im.shape[1]-1);
	    y0 = np.clip(y0, 0, im.shape[0]-1);
	    y1 = np.clip(y1, 0, im.shape[0]-1);

	    Ia = im[ y0, x0 ]
	    Ib = im[ y1, x0 ]
	    Ic = im[ y0, x1 ]
	    Id = im[ y1, x1 ]

	    wa = (x1-x) * (y1-y)
	    wb = (x1-x) * (y-y0)
	    wc = (x-x0) * (y1-y)
	    wd = (x-x0) * (y-y0)

	    return wa.reshape([len(wa),1])*Ia + wb.reshape([len(wb),1])*Ib + wc.reshape([len(wc),1])*Ic + wd.reshape([len(wd),1])*Id



if __name__ == '__main__':

	test = Reprojectimage('/home/wrlife/project/deeplearning/image_reproject/oldcase1')
	test.projective_inverse_warp()