# %%
import voltools as vt
import cupy as cp
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import angle_to_angle_list
from pytom_tm.correlation import normalised_cross_correlation
from pytom_tm.matching import TemplateMatchingGPU
import numpy as np
from matplotlib import pyplot as plt

# %%
t_size = 24
#volume = np.zeros((100,) * 3, dtype=float)
volume =spherical_mask(100,10,2).sum(axis=1)
#volume =np.random.random((100,100))

template = np.zeros((t_size,) * 3, dtype=float)
template[3:8, 4:8, 3:7] = 1.0
template[7, 8, 5:7] = 1.0
template=spherical_mask(t_size,5,2)
mask = spherical_mask(t_size, (t_size/2)-1, 0.5)
gpu_id = "gpu:0"
angles = angle_to_angle_list(38.53)



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

score_volume, angle_volume, stats = tm.run()# %%

# %%
plt.imshow(score_volume, cmap='gray')
plt.colorbar();

# %%
