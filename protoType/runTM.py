# %%
import voltools as vt
import cupy as cp
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import angle_to_angle_list
from pytom_tm.correlation import normalised_cross_correlation
from pytom_tm.matching import TemplateMatchingGPU
import numpy as np
from matplotlib import pyplot as plt
from pytom_tm.io import read_mrc,write_mrc

# %%
t_size = 64
basp="/fs/pool/pool-fbeck/projects/4InsituTM/src/pytom2d"
#volume = np.zeros((100,) * 3, dtype=float)
#volume =spherical_mask(100,10,2).sum(axis=1)
#volume =np.random.random((100,100))
#volume=np.roll(volume, 15, axis=1)
volume=read_mrc(basp+'/tmpOut/img_4Rot4.mrc')
#template=spherical_mask(t_size,5,2)
template=read_mrc(basp+'/tmpOut/riboSphere_3.mrc')
mask = spherical_mask(t_size, (t_size/2)-1, 0.5)
#gpu_id = "gpu:0"
#angles=[]
#nt=np.array([30,0,0],dtype=np.float64)*3.14/180
#angles.append((nt[0],nt[1],nt[2]))
#
# angles=[]
# anglesDeg=[]
# for i in range(0,360,10):
#     nt=np.array([i,50,30],dtype=np.float64)*3.14/180
#     angles.append((nt[0],nt[1],nt[2])) 
#     ntDeg=np.array([i,50,30],dtype=np.float64)
#     anglesDeg.append((ntDeg[0],ntDeg[1],ntDeg[2]))
angles=angle_to_angle_list(8)
# nt=np.array([29,0,0],dtype=np.float64)*3.14/180
# angles.append((nt[0],nt[1],nt[2]))
# nt=np.array([20,0,0],dtype=np.float64)*3.14/180
# angles.append((nt[0],nt[1],nt[2]))

# nt=np.array([25,0,0],dtype=np.float64)*3.14/180
# angles.append((nt[0],nt[1],nt[2]))
# nt=np.array([31,0,0],dtype=np.float64)*3.14/180
# angles.append((nt[0],nt[1],nt[2]))
# nt=np.array([29,0,0],dtype=np.float64)*3.14/180
# angles.append((nt[0],nt[1],nt[2]))
# nt=np.array([15,0,0],dtype=np.float64)*3.14/180
# angles.append((nt[0],nt[1],nt[2]))



# %%

tm = TemplateMatchingGPU(
            0,
            0,
            volume,
            template,
            mask,
            angles,
            list(range(len(angles))),
        )

score_volume, angle_volume, cc_map,stats = tm.run()# %%

# %%
ccval=score_volume.max()
pos=np.unravel_index(score_volume.argmax(),score_volume.shape)
angIdx=angle_volume[pos[0],pos[1]]
print(pos[0],pos[1],ccval)
#print("scoresMax:" + str(score_volume.max()))
ang=angles[int(angIdx)]
angDeg=np.array(ang)*180/3.14
print(angIdx,angDeg)

# %%
templatecp=cp.asarray(template, dtype=cp.float32, order="C")
template_texture= vt.StaticVolume(
            templatecp, interpolation="filt_bspline", device=f"gpu:{0}"
        )

tmpRot=cp.asarray(template, dtype=cp.float32, order="C")
template_texture.transform(
                rotation=(ang[0], ang[1], ang[2]),
                rotation_order="rzxz",
                output=tmpRot,
                rotation_units="rad",
            )

tmpRot=tmpRot.sum(axis=2)



# %%


fig, axs = plt.subplots(1, 4)


axs[0].imshow(template[:,:,33], cmap='gray')
axs[0].set_aspect('equal')

axs[1].imshow(tmpRot.get(), cmap='gray')
axs[1].set_aspect('equal')

#fig.colorbar(im1,ax=axs[0]);
axs[2].imshow(volume, cmap='gray')
axs[2].scatter(pos[1],pos[0], color='red', marker='x')
axs[2].set_aspect('equal')

im1=axs[3].imshow(cc_map, cmap='gray')
axs[3].set_title('CC Max Val:'+ str(score_volume.max()))
axs[3].scatter(pos[1],pos[0], color='red', marker='x')
axs[3].set_aspect('equal')




# %%
