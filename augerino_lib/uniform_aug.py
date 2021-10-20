'''Extension (with our modification) from the original Augerino code https://github.com/g-benton/learning-invariances'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from augerino_lib.augerino_utils import expm
import torch.distributions as D

class UniformAug(nn.Module):
    """docstring for MLPAug"""
    def __init__(self, trans_scale=0.1):
        super(UniformAug, self).__init__()

        self.trans_scale = trans_scale

        self.width = nn.Parameter(torch.zeros(6))
        self.softplus = torch.nn.Softplus()
        self.g0 = None
        self.std_batch_size = None

    def set_width(self, vals):
        self.width.data = vals
        
    def transform(self, x):
        bs, _, w, h = x.size()
        weights = torch.rand(bs, 6)
        weights = weights.to(x.device, x.dtype)
        width = self.softplus(self.width)
        weights = weights * width - width.div(2.)
        generators = self.generate(weights)

        ## exponential map
        affine_matrices = expm(generators)
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size = x.size(),
                                 align_corners=True)
        x_out = F.grid_sample(x, flowgrid,align_corners=True)
        return x_out

    def generate(self, weights):
        """
        return the sum of the scaled generator matrices
        """
        bs = weights.shape[0]

        if self.g0 is None or self.std_batch_size != bs:
            self.std_batch_size = bs

            ## tx
            self.g0 = torch.zeros(3, 3, device=weights.device)
            self.g0[0, 2] = 1. * self.trans_scale
            self.g0 = self.g0.unsqueeze(-1).expand(3,3, bs)

            ## ty
            self.g1 = torch.zeros(3, 3, device=weights.device)
            self.g1[1, 2] = 1. * self.trans_scale
            self.g1 = self.g1.unsqueeze(-1).expand(3,3, bs)

            self.g2 = torch.zeros(3, 3, device=weights.device)
            self.g2[0, 1] = -1.
            self.g2[1, 0] = 1.
            self.g2 = self.g2.unsqueeze(-1).expand(3,3, bs)

            self.g3 = torch.zeros(3, 3, device=weights.device)
            self.g3[0, 0] = 1.
            self.g3[1, 1] = 1.
            self.g3 = self.g3.unsqueeze(-1).expand(3,3, bs)

            self.g4 = torch.zeros(3, 3, device=weights.device)
            self.g4[0, 0] = 1.
            self.g4[1, 1] = -1.
            self.g4 = self.g4.unsqueeze(-1).expand(3,3, bs)

            self.g5 = torch.zeros(3, 3, device=weights.device)
            self.g5[0, 1] = 1.
            self.g5[1, 0] = 1.
            self.g5 = self.g5.unsqueeze(-1).expand(3,3, bs)

        out_mat = weights[:, 0] * self.g0
        out_mat += weights[:, 1] * self.g1
        out_mat += weights[:, 2] * self.g2
        out_mat += weights[:, 3] * self.g3
        out_mat += weights[:, 4] * self.g4
        out_mat += weights[:, 5] * self.g5

        # transposes just to get everything right
        return out_mat.transpose(0, 2).transpose(2, 1)

    def forward(self, x):
        return self.transform(x)

class MyUniformAug(nn.Module):
    """docstring for MLPAug"""
    def __init__(self, transfos = ['tx', 'ty', 'rot', 'scale', 'strech', 'shear'], 
                shutvals = [1,1,1,1,1,1],
                num_classes=1):
        super(MyUniformAug, self).__init__()

        self.transfos = transfos
        self.shutvals = nn.Parameter(data=torch.FloatTensor(shutvals),
         requires_grad=False)
        self.n_transfos = len(transfos)
        self.width = nn.Parameter(torch.zeros((num_classes,self.n_transfos)))
        self.softplus = torch.nn.Softplus()
        self.create_generators(1)
        print("Init Uniform Aug model")
        print("Created generators")
        
    def set_width(self, vals):
        self.width.data = vals
        
    def transform(self, x, y=0):
        bs, _, w, h = x.size()
        weights = torch.rand(bs, self.n_transfos)
        weights = weights.to(x.device, x.dtype)
        width = self.softplus(self.width[y,:]) #take the weights that corresponds to the class (class is 0 if not fed)
        weights = weights * width - width.div(2.)
        generators = self.generate(weights)
        ## exponential map
        affine_matrices = expm(generators)
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size = x.size(),
                                 align_corners=True)
        x_out = F.grid_sample(x, flowgrid,align_corners=True)
        return x_out

    def max_transform(self, x, y=0):
        bs, _, w, h = x.size()
        weights = torch.ones(bs, self.n_transfos) #take 1
        weights = weights.to(x.device, x.dtype)
        width = self.softplus(self.width[y,:])
        weights = weights * width - width.div(2.)
        generators = self.generate(weights)
        ## exponential map
        affine_matrices = expm(generators)
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size = x.size(),
                                 align_corners=True)
        x_out = F.grid_sample(x, flowgrid,align_corners=True)
        return x_out

    def create_generators(self, bs):
        g = []
        ## tx
        if 'tx' in self.transfos:
            print("Using tx")
            gx = torch.zeros(3, 3)
            gx[0, 2] = 1. 
            gx = gx.unsqueeze(-1).unsqueeze(-1).expand(3,3, bs, 1)
            g.append(gx)

        ## ty
        if 'ty' in self.transfos:
            print("Using ty")
            gy = torch.zeros(3, 3)
            gy[1, 2] = 1. 
            gy = gy.unsqueeze(-1).unsqueeze(-1).expand(3,3, bs, 1)
            g.append(gy)

        if 'rot' in self.transfos:
            print("Using rot")
            gr = torch.zeros(3, 3)
            gr[0, 1] = -1.
            gr[1, 0] = 1.
            gr = gr.unsqueeze(-1).unsqueeze(-1).expand(3,3, bs, 1)
            g.append(gr)

        if 'scale' in self.transfos:
            print("Using scale")
            gs = torch.zeros(3, 3) 
            gs[0, 0] = 1. 
            gs[1, 1] = 1. 
            gs = gs.unsqueeze(-1).unsqueeze(-1).expand(3,3, bs, 1)
            g.append(gs)

        if 'strech' in self.transfos:
            print("Using strech")
            gst = torch.zeros(3, 3)
            gst[0, 0] = 1.
            gst[1, 1] = -1.
            gst = gst.unsqueeze(-1).unsqueeze(-1).expand(3,3, bs, 1)
            g.append(gst)

        if 'shear' in self.transfos:
            print("Using shear")
            gsh = torch.zeros(3, 3) 
            gsh[0, 1] = 1.
            gsh[1, 0] = 1.
            gsh = gsh.unsqueeze(-1).unsqueeze(-1).expand(3,3, bs, 1)
            g.append(gsh)
            
        g_matrices = torch.cat(g, dim = -1)
        self.g_matrices = nn.Parameter(data=g_matrices, requires_grad=False)

    def generate(self, weights):
        """
        return the sum of the scaled generator matrices
        """
        out_mat = weights[:, 0] * self.g_matrices[:,:,:,0]
        for i in range(1,self.n_transfos):
            out_mat += weights[:, i] * self.g_matrices[:,:,:,i]

        # transposes just to get everything right
        return out_mat.transpose(0, 2).transpose(2, 1)

    def forward(self, x, y):
        return self.transform(x, y)

class AugModuleMin(MyUniformAug):
    """docstring for AugModule"""
    def __init__(self, transfos = ['tx', 'ty', 'rot', 'scale', 'strech', 'shear'],
                shutvals =  [1,1,1,1,1,1],
                min_values = [[1, 1, 1, 1, 1, 1]],
                num_classes = 1):
        super(AugModuleMin, self).__init__(transfos=transfos, shutvals=shutvals, num_classes = num_classes)
        min_values = torch.FloatTensor(min_values)
        if len(min_values.size())==1:
            min_values = min_values.unsqueeze(0)
        self.min_values = nn.Parameter(min_values, requires_grad=False)
        assert self.min_values.size(1)==len(transfos)

    def transform(self, x, y=0):
        bs, _, _, _ = x.size()
        weights = torch.rand(bs, self.n_transfos)
        weights = weights.to(x.device, x.dtype)
        widthsp = self.softplus(self.width[y,:])
        width = torch.maximum(widthsp, self.min_values[y,:])
        weights = weights * width - width.div(2.)
        generators = self.generate(weights)
        ## exponential map
        affine_matrices = expm(generators)
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size = x.size(),
                                 align_corners=True)
        x_out = F.grid_sample(x, flowgrid,align_corners=True)
        return x_out

    def max_transform(self, x, y=0):
        bs, _, _, _ = x.size()
        weights = torch.ones(bs, self.n_transfos) # take 1
        weights = weights.to(x.device, x.dtype)
        widthsp = self.softplus(self.width[y,:])
        width = torch.maximum(widthsp, self.min_values[y,:])
        weights = weights * width - width.div(2.)
        generators = self.generate(weights)
        ## exponential map
        affine_matrices = expm(generators)
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size = x.size(),
                                 align_corners=True)
        x_out = F.grid_sample(x, flowgrid, align_corners=True)
        return x_out

class UniformAugEachPos(MyUniformAug):
    """docstring for MLPAug"""
    def __init__(self, transfos = ['tx', 'ty', 'rot', 'scale', 'strech', 'shear'], 
                shutvals =  [1,1,1,1,1,1],
                num_classes=1):
        super(UniformAugEachPos, self).__init__(transfos=transfos,shutvals=shutvals,
                                num_classes=num_classes)
    
    def transform(self, x, y=0):
        bs, _, w, h = x.size()
        weights = torch.rand(bs, self.n_transfos) 
        weights_t = weights[:,:2] # for translation
        weight_s = weights[:,2][:,None] # for scale 
        weights_t = weights_t.to(x.device, x.dtype)
        weight_s = weight_s.to(x.device, x.dtype)
        width = self.softplus(self.width[y,:])
        width_t = width[:2]
        width_s = width[2]
        weights_t = weights_t * width_t - width_t.div(2.)
        weight_s = weight_s * width_s.div(2) - width_s.div(2)
        weights = torch.cat([weights_t,weight_s],dim=-1)
        x_out = x
        for i in range(self.n_transfos):
            out_mat_i = weights[:, i] * self.g_matrices[:,:,:,i]
            out_mat_i = out_mat_i.transpose(0, 2).transpose(2, 1)
            ## exponential map
            affine_matrix = expm(out_mat_i)
            flowgrid = F.affine_grid(affine_matrix[:, :2, :], size = x_out.size(),
                                 align_corners=True)
            x_out = F.grid_sample(x_out, flowgrid,align_corners=True)
        return x_out

    def max_transform(self, x, y=0):
        bs, _, w, h = x.size()
        weights = torch.ones(bs, self.n_transfos) 
        weights_t = weights[:,:2] # for translation
        weight_s = weights[:,2][:,None] # for scale 
        weights_t = weights_t.to(x.device, x.dtype)
        weight_s = weight_s.to(x.device, x.dtype)
        width = self.softplus(self.width[y,:])
        width_t = width[:2]
        width_s = width[2]
        weights_t = weights_t * width_t - width_t.div(2.)
        weight_s = weight_s * width_s.div(2) - width_s.div(2)
        weights = torch.cat([weights_t,weight_s],dim=-1)
        x_out = x
        for i in range(self.n_transfos):
            out_mat_i = weights[:, i] * self.g_matrices[:,:,:,i]
            out_mat_i = out_mat_i.transpose(0, 2).transpose(2, 1)
            ## exponential map
            affine_matrix = expm(out_mat_i)
            flowgrid = F.affine_grid(affine_matrix[:, :2, :], size = x_out.size(),
                                 align_corners=True)
            x_out = F.grid_sample(x_out, flowgrid, align_corners=True)
        return x_out

class UniformAugEachMin(AugModuleMin):
    """docstring for MLPAug"""
    def __init__(self, transfos = ['tx', 'ty', 'rot', 'scale', 'strech', 'shear'], 
                shutvals =  [1,1,1,1,1,1],
                min_values = [[1, 1, 1, 1, 1, 1]],
                num_classes=1):
        super(UniformAugEachMin, self).__init__(transfos=transfos,shutvals=shutvals,
                    min_values=min_values, num_classes=num_classes)
    
    def transform(self, x, y=0):
        bs, _, w, h = x.size()
        weights = torch.rand(bs, self.n_transfos)
        weights = weights.to(x.device, x.dtype)
        widthsp = self.softplus(self.width[y,:])
        width = torch.maximum(widthsp, self.min_values[y,:])
        weights = weights * width - width.div(2.)
        x_out = x
        for i in range(self.n_transfos):
            out_mat_i = weights[:, i] * self.g_matrices[:,:,:,i]
            out_mat_i = out_mat_i.transpose(0, 2).transpose(2, 1)
            ## exponential map
            affine_matrix = expm(out_mat_i)
            flowgrid = F.affine_grid(affine_matrix[:, :2, :], size = x_out.size(),
                                 align_corners=True)
            x_out = F.grid_sample(x_out, flowgrid,align_corners=True)
        return x_out

    def max_transform(self, x, y=0):
        bs, _, w, h = x.size()
        weights = torch.ones(bs, self.n_transfos) #take 1
        weights = weights.to(x.device, x.dtype)
        widthsp = self.softplus(self.width[y,:])
        width = torch.maximum(widthsp, self.min_values[y,:])
        weights = weights * width - width.div(2.)
        x_out = x
        for i in range(self.n_transfos):
            out_mat_i = weights[:, i] * self.g_matrices[:,:,:,i]
            out_mat_i = out_mat_i.transpose(0, 2).transpose(2, 1)
            ## exponential map
            affine_matrix = expm(out_mat_i)
            print(i,affine_matrix)
            flowgrid = F.affine_grid(affine_matrix[:, :2, :], size = x_out.size(),
                                 align_corners=True)
            x_out = F.grid_sample(x_out, flowgrid, align_corners=True)
        return x_out


class AugModuleNotCentered(MyUniformAug):
    """docstring for AugModule"""
    def __init__(self, transfos = ['tx', 'ty', 'rot', 'scale', 'strech', 'shear'],
                    num_classes = 1):
        super(AugModuleNotCentered, self).__init__(transfos, num_classes)

        self.min_widths = nn.Parameter(torch.zeros((num_classes, self.n_transfos)))
        self.max_widths = nn.Parameter(torch.zeros((num_classes, self.n_transfos)))

    def transform(self, x, y=0):
        bs, _, _, _ = x.size()
        weights = torch.rand(bs, self.n_transfos)
        weights = weights.to(x.device, x.dtype)
        min_width = self.softplus(self.min_width[y,:]).div(2.)
        max_width = self.softplus(self.max_width[y,:]).div(2.)
        weights = weights * (max_width-min_width) + min_width
        generators = self.generate(weights)
        ## exponential map
        affine_matrices = expm(generators)
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size = x.size(),
                                 align_corners=True)
        x_out = F.grid_sample(x, flowgrid,align_corners=True)
        return x_out

    def max_transform(self, x, y=0):
        bs, _, _, _ = x.size()
        weights = torch.ones(bs, self.n_transfos) #take 1
        weights = weights.to(x.device, x.dtype)
        min_width = self.softplus(self.min_width[y,:]).div(2.)
        max_width = self.softplus(self.max_width[y,:]).div(2.)
        weights = weights * (max_width-min_width) + min_width
        generators = self.generate(weights)
        ## exponential map
        affine_matrices = expm(generators)
        flowgrid = F.affine_grid(affine_matrices[:, :2, :], size = x.size(),
                                 align_corners=True)
        x_out = F.grid_sample(x, flowgrid,align_corners=True)
        return x_out
    

class UniformAugEachPosUnif(MyUniformAug):
    """docstring for MLPAug"""
    def __init__(self, transfos = ['tx', 'ty', 'rot', 'scale', 'strech', 'shear'], 
                shutvals =  [1,1,1,1,1,1],
                num_classes=1):
        super(UniformAugEachPosUnif, self).__init__(transfos=transfos,shutvals=shutvals,
                                num_classes=num_classes)
    
    def transform(self, x, y=0):
        bs, _, w, h = x.size()
        # weights = torch.rand(bs, self.n_transfos) 
        weights = torch.ones(bs, self.n_transfos) 
        weights_t = weights[:,:2] # for translation
        weight_s = weights[:,2][:,None] # for scale 
        weights_t = weights_t.to(x.device, x.dtype)
        weight_s = weight_s.to(x.device, x.dtype)
        width = self.softplus(self.width[y,:])
        width_t = width[:2]
        width_s = width[2]
        weights_t = weights_t * width_t - width_t.div(2.)
        weight_s = weight_s * width_s
        weight_s = torch.log(weight_s)
        weights = torch.cat([weights_t,weight_s],dim=-1)
        x_out = x
        for i in range(self.n_transfos):
            out_mat_i = weights[:, i] * self.g_matrices[:,:,:,i]
            out_mat_i = out_mat_i.transpose(0, 2).transpose(2, 1)
            ## exponential map
            affine_matrix = expm(out_mat_i)
            flowgrid = F.affine_grid(affine_matrix[:, :2, :], size = x_out.size(),
                                 align_corners=True)
            x_out = F.grid_sample(x_out, flowgrid,align_corners=True)
        return x_out