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
volume=read_mrc(basp+'/tmpOut/img_1.mrc')
#template=spherical_mask(t_size,5,2)
template=read_mrc(basp+'/tmpOut/riboSphere_3.mrc')
mask = spherical_mask(t_size, (t_size/2)-1, 0.5)
#gpu_id = "gpu:0"
#angles = angle_to_angle_list(38.53)
nt=np.array([0,0,0],dtype=np.float64)
angles=[]
angles.append((nt[0],nt[1],nt[2]))


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


fig, axs = plt.subplots(1, 2)
im1=axs[0].imshow(cc_map, cmap='gray')
axs[0].set_title('CC Max Val:'+ str(cc_map.max()))
axs[0].set_aspect('equal')
#fig.colorbar(im1,ax=axs[0]);
axs[1].imshow(volume, cmap='gray')
axs[1].set_aspect('equal')


# %%
