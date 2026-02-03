import torch
import torch.nn as nn
from einops import rearrange
from net.transformer_utils import *
from net.vmamba import *


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)
    

class IG_CA(nn.Module):                                # 这里现在是纯粹的通道注意力,
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)            # 这个线性层实现的 to_q和to_k、to_v 就是将特征沿通道 将其从 dim 变为 dim_head * heads
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )




    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        # bi,hi,wi,ci = illu_fea_trans.shape
        x = x_in.reshape(b, h * w, c)               
        q_inp = self.to_q(x)                       

        ill = illu_fea_trans.reshape(b, h * w, c)   
        k_inp = self.to_k(ill)                      
        v_inp = self.to_v(ill)                      
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp))

        q = q.transpose(-2, -1)                     # (b, heads,c/h, hw)
        k = k.transpose(-2, -1)                     # (b, heads,c/h, hw)
        v = v.transpose(-2, -1)                     # (b, heads,c/h, hw)
        q = F.normalize(q, dim=-1, p=2)            
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q          # Q- (b, heads,c/h, hw) --> (b, heads,hw,c/h);(b, heads,c/h, hw) @ (b, heads,hw,c/h) --> (b,heads,c/h,c/h)

        attn = attn * self.rescale          # 
        attn = attn.softmax(dim=-1)         # 
        x = attn @ v   # b,heads,d,hw                               # (b,heads,c/h,c/h) @ (b, heads,c/h,hw) --> (b,heads,c/h,hw)
        x = x.permute(0, 3, 1, 2)    # Transpose                    # (b, hw, heads, c/h)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)     # (b, hw, heads, c/h) --> (b, hw, heads * c/h) 即 (b, hw, c)
        out_c = self.proj(x).view(b, h, w, c)                       # (b,hw,c) --> (b,h,w,c)    
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(     
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out1 = out_c + out_p                                        

        return out1
    

class IG_SA(nn.Module):                               
    def __init__(
        self,
        dim,

        # hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        # ssm_d_state: int = 16,
        # ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
    ):
        super().__init__()
        self.dim = dim

        self.hidden_dim = dim


        self.conv1 = nn.Conv2d(dim*2, dim, 1, 1, 0)

        self.norm = norm_layer(self.hidden_dim)

        self.drop_path = DropPath(drop_path)


        self.op = SS2D(
                d_model=self.hidden_dim, 
                d_state=16, 
                ssm_ratio=2,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )




    def forward(self, x_in, illu_fea):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        x = self.conv1(torch.cat((x_in, illu_fea), dim=1))

        x_norm = self.norm(x.permute(0, 2, 3, 1).contiguous())


        x_global = self.drop_path(self.op(x_norm)).permute(0, 3, 1, 2).contiguous()


        return x_global


class CCAM(nn.Module):                              
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2
            # FFT = False
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_CA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim)),
                nn.Conv2d(dim, dim * 2, 1, 1, 0),
                nn.Conv2d(dim, dim, 1, 1, 0)
            ]))
        self.process1 = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * dim, dim, 1, 1, 0))
        
        self.brightness_estimator = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),                        

            nn.Conv2d(dim, dim//8, 1),          #

            nn.GELU(),
            nn.Conv2d(dim//8, 1, 1),          # 

        )

        self.scale = nn.Parameter(torch.tensor(0.5))       

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff,conv2,conv1) in self.blocks:


            x_1 = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) 
            x_1 = conv2(x_1.permute(0, 3, 1, 2))
            x1,x2 = x_1.chunk(2, dim=1)   
            x_3 = x1*x2
            x = conv1(x_3) + x.permute(0, 3, 1, 2)

            brightness_raw = self.brightness_estimator(illu_fea).squeeze(-1) 
            
            brightness_level = torch.sigmoid(brightness_raw * self.scale) * 7 + 1  # [B]
            
            L = torch.round(brightness_level)                   
            L = torch.clamp(L, 1, 4)                             

            L_continuous = brightness_level + (L - brightness_level).detach()  
            
            N = 5 - L                                             # [B]
            N = N.int()                                           # 

            _, _, H, W = x.shape
            x_freq = torch.fft.rfft2(x, norm='backward')
            mag = torch.abs(x_freq)
            pha = torch.angle(x_freq)

            for _ in range(N.max().item()):                       
                mask = (N > _).float().view(-1,1,1,1)            
                mag = mag * (1 - mask) + self.process1(mag) * mask

            real = mag * torch.cos(pha)
            imag = mag * torch.sin(pha)
            x_out = torch.complex(real, imag)
            out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

            x = x.permute(0, 2, 3, 1)

            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class CSAM(nn.Module):                             
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2
            # FFT = False
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_SA(dim=dim),
                PreNorm(dim, FeedForward(dim=dim)),
                nn.Conv2d(dim, dim * 2, 1, 1, 0),
                nn.Conv2d(dim, dim, 1, 1, 0)
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (attn, ff,conv2,conv1) in self.blocks:

            x_1 = attn(x, illu_fea) 

            x_1 = conv2(x_1)
            x1,x2 = x_1.chunk(2, dim=1)   
            x_3 = x1*x2

            x = conv1(x_3) + x
            x = ff(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x
        out = x
        return out
