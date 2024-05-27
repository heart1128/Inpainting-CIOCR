import math
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


    
class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_emb = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        _, _, N = x.shape
        x = x + F.interpolate(self.pos_emb, size=(N), mode='linear', align_corners=False)
        return x

'''
dim  =  64
key_dim =  12
num_heads =  4
attn_ratio =  2
activation =  <class 'torch.nn.modules.activation.ReLU6'>
norm_cfg =  {'type': 'BN', 'requires_grad': True}
'''
# n_embd == key_dim * num_heads  , num_head == n_head
class Sea_Attention(torch.nn.Module):
    def __init__(self, dim=164, key_dim=12, num_heads=4,
                 attn_ratio=2,
                 activation=nn.ReLU6,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        
#         print('dim=', dim, '\nkey_dim=', key_dim, '\nnum_heads=', num_heads,
#                  '\nattn_ratio', attn_ratio,
#                  '\nactivation', activation,
#                  '\nnorm_cfg=',norm_cfg )
        
        # 注意力头数量
        self.num_heads = num_heads
        # 缩放尺度
        self.scale = key_dim ** -0.5
        # 
        self.key_dim = key_dim
        # 总维度，头*key维度
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        # 
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        
        self.dwconv = Conv2d_BN(2*self.dh, 2*self.dh, ks=3, stride=1, pad=1, dilation=1,
                 groups=2*self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2*self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape


        q = self.to_q(x) # q : [B,nk_kd, W, H]
        k = self.to_k(x) # k : [B,nk_kd, W, H]
        v = self.to_v(x) # v : [B,dh, W, H]
        
        # detail enhance
        qkv = torch.cat([q,k,v],dim=1)
        qkv = self.act(self.dwconv(qkv)) # [B, 2*nk_kd + dh, W, H]
        qkv = self.pwconv(qkv) # [B , dim, W, H]

        # squeeze axial attention
        
        ## squeeze ro
        # [B, num_heads, W, k_dim]
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        # [B, num_heads, k_dim, W]
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        # [B, num_heads, W, k_dim*2]
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)

        # [B, num_heads, W, W]
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        
        ## squeeze column
        # [B, num_heads, H, k_dim]
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv
        return xx




if __name__ == '__main__':
    model_cfgs = dict(
        cfg1=[
            # k,  t,  c, s
            [3, 1, 16, 1],  
            [3, 4, 16, 2],  
            [3, 3, 16, 1]],  
        cfg2=[
            [5, 3, 32, 2],  
            [5, 3, 32, 1]],  
        cfg3=[
            [3, 3, 64, 2],  
            [3, 3, 64, 1]],
        cfg4=[
            [5, 3, 128, 2]],
        cfg5=[
            [3, 6, 160, 2]],
        channels=[16, 16, 32, 64, 128, 160],
        num_heads=4,
        emb_dims=[64, 128, 160],
        key_dims=[12, 16,24],
        depths=[2, 2, 2],
        drop_path_rate=0.1,
        mlp_ratios=[2,4, 4]
    )
    # model = SeaFormer(
    #     cfgs=[model_cfgs['cfg1'], model_cfgs['cfg2'], model_cfgs['cfg3'], model_cfgs['cfg4'], model_cfgs['cfg5']],
    #     channels=model_cfgs['channels'],
    #     key_dims=model_cfgs['key_dims'],
    #     emb_dims=model_cfgs['emb_dims'],
    #     depths=model_cfgs['depths'],
    #     num_heads=model_cfgs['num_heads'],
    #     mlp_ratios=model_cfgs['mlp_ratios'],
    #     drop_path_rate=model_cfgs['drop_path_rate'])

    input = torch.rand((12, 256, 32, 32))
    attn = Sea_Attention(dim=256, key_dim=256 // 8, num_heads=8)
    output = attn(input)
    print(output.shape)

    # print(model)

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    attn.eval()
    flops = FlopCountAnalysis(attn, input)
    print(flop_count_table(flops))

