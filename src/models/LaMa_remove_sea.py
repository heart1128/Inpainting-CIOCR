import numpy as np

from .ffc import *
from .layers import *
from .sea_attention import Sea_Attention as SEA_Attention

from timm.models.layers import trunc_normal_
import math


class ResnetBlock_remove_IN(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1, stride=1, groups=1, dilation=1):
        super(ResnetBlock_remove_IN, self).__init__()

        self.ffc1 = FFC_BN_ACT(dim, dim, kernel_size, 0.75, 0.75, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)

        self.ffc2 = FFC_BN_ACT(dim, dim, kernel_size, 0.75, 0.75, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)

    def forward(self, x):
        output = x
        _, c, _, _ = output.shape
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)
        x_l, x_g = self.ffc1(output)
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        output = x + output

        return output

"""
F : frequency domain
A : airspace
F : fusion

"""
class AttFAF(nn.Module):
    
    def __init__(self, ngf, kernel_size=3, padding=1, stride=1, groups=1):
        super(AttFAF, self).__init__()

        # print(ngf, groups)
        
        # FFC and Sigmod, 不要全局输出
        self.mask = FFC_BN_ACT(ngf, 1, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, 
                    norm_layer=nn.BatchNorm2d, activation_layer=nn.Sigmoid,
                    **{"ratio_gin": 0.75, "ratio_gout": 0, "enable_lfu": False})
        
        # 连接之后的FFC ，不要全局FFT
        self.minus = FFC_BN_ACT(ngf+1, ngf, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                           **{"ratio_gin": 0, "ratio_gout": 0.75, "enable_lfu": False})

        # 最后相加的FFC
        self.add = FFC_BN_ACT(ngf, ngf, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                           norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, 
                           **{"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False})
        
        

    def forward(self, x):
        # x_l, x_g = x if type(x) is tuple else (x, 0)

        [b, c, h, w] = x.shape
        x_l, x_g = torch.split(x, [c - int(c * 0.75), int(c * 0.75)], dim=1)

        
        # dim = 1
        mask, _ = self.mask((x_l, x_g))
        
        
        minus_l, minus_g = self.minus(torch.cat([x_l, x_g, mask], 1))

        add_l, add_g = self.add((x_l - minus_l, x_g - minus_g))

        x_l, x_g = x_l - minus_l + add_l, x_g - minus_g + add_g

        output = torch.cat([x_l, x_g], dim=1)

        return output


"""  SAM块  """
class SAMFFC(nn.Module):
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None, 
                       attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        # print(dim, ca_num_heads, sa_num_heads, qkv_bias, qk_scale, 
        #                attn_drop, proj_drop, ca_attention, expand_ratio)

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups=self.dim // ca_num_heads

        if ca_attention == 1:
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)  
            for i in range(self.ca_num_heads):
                '''  不使用普通卷积，使用改良的FFCAttn  '''
                # local_conv = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.ca_num_heads)
                local_conv = ResnetBlock_remove_IN(dim//ca_num_heads, kernel_size=(3+i*2), padding=(i+1), stride=1, groups=dim//self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
            self.proj0 = nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups)
            self.bn = nn.BatchNorm2d(dim*expand_ratio)
            self.proj1 = nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H, W):

        B, N, C = x.shape
        ''' 这个是多尺度token注意力 ''' 
        if self.ca_attention == 1:
            v = self.v(x)
            # [分段数，B, 分段长度，H,W]
            s = self.s(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2)
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i = s[i] # 拿出一段进行卷积
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W) # 分成[B，组数，每组的数量, H, W]
                if i == 0:
                    s_out = s_i  # 第一组不用拼接
                else:
                    s_out = torch.cat([s_out,s_i],2) # 按组数量组合成groups
            s_out = s_out.reshape(B, C, H, W) # 所有groups组合就是C， 每次卷积都会从C中拿出一个通道，所以每个都进行了1*1融合
            s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
            self.modulator = s_out
            s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
            x = s_out * v
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    



class ResnetSeaAttention(nn.Module):
    def __init__(self, dim):
        super(ResnetSeaAttention, self).__init__()
        self.sea = SEA_Attention(dim, dim // 8, 8)
    
    def forward(self, x):
        output = x
        x = self.sea(x)
        output = output + x
        return output


class MaskedSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        """`input_ids` is expected to be [bsz x seqlen]."""
        return super().forward(input_ids)


class MultiLabelEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_positions, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input_ids):
        # input_ids:[B,HW,4](onehot)
        out = torch.matmul(input_ids, self.weight)  # [B,HW,dim]
        return out


class LaMa_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # self.sea_blocks = nn.ModuleList()
        # ### attn
        # for i in range(3):  
        #     cur_resattn = ResnetSeaAttention(512)
        #     self.sea_blocks.append(cur_resattn)

        ### AttnFFC
        self.attnFFC_blocks = nn.ModuleList()
        for _ in range(3):
            cur_block = AttFAF(512)
            self.attnFFC_blocks.append(cur_block)



        self.sam_blocks = nn.ModuleList()
        ### resnet blocks
        for i in range(3): 
            cur_resblock = SAMFFC(512, 4, 8, True, None, 0.0, 0.0, 1, 2)
            # cur_resblock = AttFAF(256)
            self.sam_blocks.append(cur_resblock)
    

        # self.middle = nn.Sequential(*blocks)

        self.convt1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt1 = nn.BatchNorm2d(256)

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt2 = nn.BatchNorm2d(128)

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt3 = nn.BatchNorm2d(64)

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)
        self.act_last = nn.Tanh()

    def forward(self, x, rel_pos_emb=None, direct_emb=None, str_feats=None):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        for i in range(3):
            x = self.attn_blocks[i](x)

        x = self.middle(x)

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        return x


# # FCU耦合单元
# class FCU_Conv_To_Toekn(nn.Module):
#     def __init__(self, in_channels, x_size):
#         super().__init__()

#         # 1*1 特征聚合
#         self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         # 适应归一化
#         self.LN = nn.LayerNorm([in_channels, x_size[0], x_size[1]])
#         self.act = nn.GELU()

#     def forward(self, x):
#         residual = x
#         x = self.fusion_conv(x)
#         x = self.LN(x)
#         x = self.act(x)

#         return x + residual


# class FCU_Toekn_To_Conv(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()

#         # 1*1 特征聚合
#         self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         # 适应归一化
#         self.BN = nn.BatchNorm2d(in_channels)
#         self.act = nn.GELU()


#     def forward(self, x):
#         residual = x
#         x = self.fusion_conv(x)
#         x = self.BN(x)
#         x = self.act(x)

#         return x + residual

class ReZeroFFC(LaMa_model):
    def __init__(self, config):
        super().__init__()
        self.config = config
 
        # self.FCUDown = FCU_Conv_To_Toekn(512, (32, 32))
        # self.FCUUp = FCU_Toekn_To_Conv(512)

    def forward(self, x, rel_pos_emb=None, direct_emb=None, str_feats=None):

        # if str_feats is None:
        #     str_feats = []
        #     str_feats.append(torch.rand([9, 64, 256, 256]))
        #     str_feats.append(torch.rand([9, 128, 128, 128]))
        #     str_feats.append(torch.rand([9, 256, 64, 64]))
        #     str_feats.append(torch.rand([9, 512, 32, 32]))
        #     str_feats.append(torch.rand([9, 512, 32, 32]))


        x = self.pad1(x)
        x = self.conv1(x)
        if self.config.use_MPE:
            inp = x.to(torch.float32) + rel_pos_emb + direct_emb
        else:
            inp = x.to(torch.float32)
        x = self.bn1(inp)
        x = self.act(x)

        x = self.conv2(x + str_feats[0])  # 256 cnn
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x + str_feats[1]) # 128 cnn
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x + str_feats[2]) # 64 cnn
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        # 混合特征 (sea -> attnFFC -> samFFC) * 3
        for i in range(len(self.sam_blocks)):
            B, C ,H, W = x.shape

            # list不会放到gpu中
            # print(next(self.sea_blocks[i].parameters()).device)
            # print(next(self.parameters()).device)
            # print(next(self.attnFFC_blocks[i].parameters()).device)
            if i == 0:
                # x = self.sea_blocks[i](x + str_feats[3]) # 32 cnn
                x = self.attnFFC_blocks[i](x)
                x = x + str_feats[3]            # 32 tr
                x = x.reshape(B, C, H*W).permute(0, 2, 1)
                x = self.sam_blocks[i](x, H, W)
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
            else:
                # x = self.sea_blocks[i](x)
                x = self.attnFFC_blocks[i](x)
                x = x.reshape(B, C, H*W).permute(0, 2, 1)
                x = self.sam_blocks[i](x, H, W)
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
        


        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        return x


class StructureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.rezero_for_mpe is None:
            self.rezero_for_mpe = False
        else:
            self.rezero_for_mpe = config.rezero_for_mpe

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = GateConv(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(True)

        self.conv2 = GateConv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        
        # attn_blocks
        # self.sea_attn = SEA_Attention(512, 512 // 8, 8)
        # self.alpha0 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        

        blocks = []
        # resnet blocks
        for i in range(2): # 3->2
            blocks.append(ResnetBlock(input_dim=512, out_dim=None, dilation=2))

        self.middle = nn.Sequential(*blocks)
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt1 = GateConv(512, 256, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt1 = nn.BatchNorm2d(256)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt2 = GateConv(256, 128, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt2 = nn.BatchNorm2d(128)
        self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt3 = GateConv(128, 64, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt3 = nn.BatchNorm2d(64)
        self.alpha4 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        if self.rezero_for_mpe:
            self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=config.rel_pos_num,
                                                                   embedding_dim=64)
            self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=64)
            self.alpha5 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
            self.alpha6 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, x, rel_pos=None, direct=None):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        return_feats = []
        # x = self.sea_attn(x)
        # return_feats.append(x * self.alpha0)

        x = self.middle(x)
        return_feats.append(x * self.alpha1)

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha2)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha3)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha4)
        
        # [256, 128, 64, cnn32, tr32]
        return_feats = return_feats[::-1]

        if not self.rezero_for_mpe:
            return return_feats
        else:
            b, h, w = rel_pos.shape
            rel_pos = rel_pos.reshape(b, h * w)
            rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
            direct = direct.reshape(b, h * w, 4).to(torch.float32)
            direct_emb = self.direct_emb(direct).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha6

            return return_feats, rel_pos_emb, direct_emb


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kw, stride=2, padding=padw)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kw, stride=2, padding=padw)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kw, stride=2, padding=padw)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kw, stride=1, padding=padw)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2.to(torch.float32))
        conv2 = self.act(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3.to(torch.float32))
        conv3 = self.act(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4.to(torch.float32))
        conv4 = self.act(conv4)

        conv5 = self.conv5(conv4)
        conv5 = self.bn5(conv5.to(torch.float32))
        conv5 = self.act(conv5)

        conv6 = self.conv6(conv5)

        outputs = conv6

        return outputs, [conv1, conv2, conv3, conv4, conv5]


# if __name__ == "__main__":

#     class Config:
#         def __init__(self) -> None:
#             self.use_MPE = False

#     config = Config();
#     model = ReZeroFFC(config=config)

#     from fvcore.nn import FlopCountAnalysis, flop_count_table
#     model.eval()
#     input = torch.rand((9, 4, 256, 256))
#     model(input)
#     flops = FlopCountAnalysis(model, input)
#     # print(flop_count_table(flops))

 