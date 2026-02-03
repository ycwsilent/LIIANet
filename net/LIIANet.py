import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from huggingface_hub import PyTorchModelHubMixin
from net.CCSAM import *

class LIIANet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144], 
                #  heads=[1, 2, 4, 8],
                 heads=[2, 4, 8, 8],
                 norm=False
        ):
        super(LIIANet, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels                             # 
        [head1, head2, head3, head4] = heads                        # 

        self.channels = channels
        self.heads = heads

        self.CCSAM_1 = CCAM(dim=ch1,num_blocks=1,heads=head1,dim_head=ch1//head1)          # 
        self.CCSAM_2 = CCAM(dim=ch2,num_blocks=2,heads=head2,dim_head=ch1//head2)
        self.CCSAM_b = CCAM(dim=ch3,num_blocks=2,heads=head3,dim_head=ch3//head3)
        self.CCSAM_3 = CSAM(dim=ch2,num_blocks=2,heads=head2,dim_head=ch1//head2)         # 交
        self.CCSAM_4 = CSAM(dim=ch1,num_blocks=1,heads=head1,dim_head=ch1//head1)
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),                                 # 
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)    # 
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm) # 
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm) # 
        
        # self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)   # 
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 3, 3, stride=1, padding=0,bias=False)    # 
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),                 # 
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),       # 
            )

        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)      # 
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)      # 
                
        self.trans = RGB_HVI()
        
    def forward(self, x):
        dtypes = x.dtype                                
        hvi = self.trans.HVIT(x)                        
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)        

        i_enc0 = self.IE_block0(i)                      # (b,1,h,w) ---> (b,ch1,h,w)    #
        hv_0 = self.HVE_block0(hvi)                     

        hv_0 = self.CCSAM_1(hv_0,i_enc0)                  # 

        i_enc1 = self.IE_block1(i_enc0)                 
 
        hv_1 = self.HVE_block1(hv_0)                  

        hv_1 = self.CCSAM_2(hv_1,i_enc1)                

        i_enc2 = self.IE_block2(i_enc1)                 
        hv_2 = self.HVE_block2(hv_1)                    

        hv_b = self.CCSAM_b(hv_2,i_enc2)                

        output_rgb_mid = self.trans.PHVIT(hv_b)   

        hvD_2 = self.HVD_block2(hv_b, hv_1)          

        hvD_1 = self.CCSAM_3(hvD_2,i_enc1)              #   

        hvD_1 = self.HVD_block1(hvD_1,hv_0)             #   (b,ch2,h/2,w/2),(b,ch1,h,w) --->  (b,ch1,h,w)

        hvD_0 = self.CCSAM_4(hvD_1,i_enc0)              #   (b,ch1,h,w),(b,ch1,h,w) ---> (b,ch1,h,w)


        hv_0 = self.HVD_block0(hvD_0)                   #(b,ch1,h,w) ---> (b,3,h,w)
        output_hvi = hv_0 + hvi                         # 
        output_rgb = self.trans.PHVIT(output_hvi)       # 

        return output_rgb_mid,output_rgb                # 
    
    def HVIT(self,x):                                   # 
        hvi = self.trans.HVIT(x)
        return hvi
    
    