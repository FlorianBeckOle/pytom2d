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
from pytom_tm.weights import create_ctf

# %%
t_size = 64

basp="../data/"
volume=read_mrc(basp+'img_1.mrc')
template=read_mrc(basp+'riboSphere_3.mrc')
mask=read_mrc(basp+'/maskTmpl.mrc')

maskIsSpherical=False
angles=angle_to_angle_list(12) #ang list with increment 9
pixS=1 #in Ang
defocus=0.15 #in mue here very small used to reduce sampling artifacts
ampToPhaseRat=0.08
volt=300
cs=2.7
print("sampling of ctf is too small in this example")
ctf = create_ctf([t_size,t_size], pixS * 1e-10, defocus*1e-6,ampToPhaseRat , volt*1e3, cs*1e-3)
plt.imshow(ctf)        
#ctf=None #use none to switch ctf correction off!


# %%

tm = TemplateMatchingGPU(
            0,
            0,
            volume,
            template,
            mask,
            angles,
            list(range(len(angles))),
            maskIsSpherical,
            ctf
        )

score_volume, angle_volume,stats = tm.run()# %%

# %% extract first peak
ccval=score_volume.max()
pos=np.unravel_index(score_volume.argmax(),score_volume.shape)
angIdx=angle_volume[pos[0],pos[1]]
print(pos[0],pos[1],ccval)
#print("scoresMax:" + str(score_volume.max()))
ang=angles[int(angIdx)]
angDeg=np.array(ang)*180/3.14
print(angIdx,angDeg)

# %% Rotate template around opt. angle and project
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


fig, axs = plt.subplots(2, 2)


axs[0,0].imshow(template[:,:,33], cmap='gray')
axs[0,0].set_aspect('equal')
axs[0,0].set_title("template slice",fontsize=8)

axs[0,1].imshow(tmpRot.get(), cmap='gray')
axs[0,1].set_aspect('equal')
axs[0,1].set_title("template rotated",fontsize=8)

#fig.colorbar(im1,ax=axs[0]);
axs[1,0].imshow(volume, cmap='gray')
axs[1,0].scatter(pos[1],pos[0], color='red', marker='+',s=4)
axs[1,0].set_aspect('equal')
axs[1,0].set_title("search img",fontsize=8)


im1=axs[1,1].imshow(score_volume, cmap='gray')
axs[1,1].set_title('CC Max Val:'+ str(score_volume.max()),fontsize=8)
axs[1,1].scatter(pos[1],pos[0], color='red', marker='+',s=4)
axs[1,1].set_aspect('equal')





# %%
