import jax.numpy as jnp
import jax
from flax import linen as nn 
import numpy as np
from typing import Sequence

class BACKFLOW(nn.Module):
    # features: Sequence[int] 
    num_orbital:int
    num_electron:int

    # def setup(self):
    #    self.layers = [nn.Dense(n) for n in self.features]

    @nn.compact
    def __call__(self, x):
        # for i, lyr in enumerate(self.layers):
        #   x = lyr(x)
        #   if i != len(self.layers) - 1:
        #     x = nn.relu(x) 
        # # TODO: select corresponding rows and columns of the matrix
        # MLP
        y = jnp.copy(x)
        selected_configs=[]
        for j in range(y.shape[0]):
            if y[j]==1:
                selected_configs.append(j)
                
        x = nn.Dense(features=64)(x)  
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        #Backflow
        x = nn.DenseGeneral(features=(self.num_orbital,self.num_electron))(x)
        x = x[:, selected_configs, :]
        

        
        return jnp.linalg.det(x)