"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

'''Extension (with our modification) from the original Augerino code https://github.com/g-benton/learning-invariances'''

import torch.nn as nn
import torch.nn.functional as F
import torch

class DiffAug(nn.Module):
    """ """
    def __init__(self):
        super().__init__()
        self.Sigma = nn.Parameter(torch.eye(3))
        self.Mu = nn.Parameter(torch.zeros(3))
    def translateRotate(self, x):
        bs, _, w, h = x.size()
        z = torch.randn(bs,3,device=x.device,dtype=x.dtype)@self.Sigma + self.Mu
        # Build affine matrices for random translation of each image
        affineMatrices = torch.zeros(bs,2,3,device=x.device,dtype=x.dtype)
        affineMatrices[:,0,0] = z[:,2].cos()
        affineMatrices[:,0,1] = -z[:,2].sin()
        affineMatrices[:,1,0] = z[:,2].sin()
        affineMatrices[:,1,1] = z[:,2].cos()
        affineMatrices[:,:2,2] = z[:,:2]/(.5*w+.5*h)
        affineMatrices = affineMatrices

        flowgrid = F.affine_grid(affineMatrices, size = x.size(),align_corners=True)
        x_out = F.grid_sample(x, flowgrid,align_corners=True)
        return x_out

    def forward(self, x):
        return self.translateRotate(x)

    def log_data(self,logger,step,name):
        print(self.Sigma@self.Sigma.T,self.Mu)

    def __repr__(self):
        return self.__class__.__name__

class AugAveragedModel(nn.Module):
    def __init__(self,model,aug,disabled=False,ncopies=4,onecopy=True):
        super().__init__()
        self.aug = aug
        self.model=model
        self.ncopies = ncopies
        self.disabled = disabled
        self.onecopy=onecopy

    def forward(self,x,y=0):
        
        if self.disabled:
            return self.model(x)
        else:
            if self.training and self.onecopy: 
                return self.model(self.aug(x,y))
            else: 
                #Faster batched implementation
                #return (sum(F.log_softmax(self.model(self.aug(x)),dim=-1) for _ in range(self.ncopies))/self.ncopies)#.log()
                bs = x.shape[0]
                aug_x = torch.cat([self.aug(x,y) for _ in range(self.ncopies)],dim=0)
                return sum(torch.split(F.log_softmax(self.model(aug_x),dim=-1),bs))/self.ncopies
            