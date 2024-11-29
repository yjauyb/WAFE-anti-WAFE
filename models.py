import torch
from torch import nn

class Mlp(nn.Module):
    def __init__(self, in_feature_channels, hidden_feature_channels=None, out_feature_channels=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                ):
        super().__init__()
        out_feature_channels = out_feature_channels or in_feature_channels
        hidden_feature_channels = hidden_feature_channels or in_feature_channels        
        self.norm = norm_layer(in_feature_channels)
        self.fc1 = nn.Linear(in_feature_channels, hidden_feature_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feature_channels, out_feature_channels)
    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)  
        x = self.act(x)        
        x = self.fc2(x) 
        x = shortcut+x
        return x
    
class RelevancyWeightedSum1D(nn.Module):
    """ 
    多区间相关度加权和.
    Args:        
        in_features_channels (int): number of input feature channles. Default: 256
        num_intervals (int): Number of intervals for calculate relevancy. default: 8         
        xyz_bias (bool):  If True, add a learnable bias to x, y, z in the Linear fuction before relevance calculation. Default: True.
        end_bias (bool):  If True, add a learnable bias to end Linear fuction.Default: True.
        r_scale (float | "sqr"): If float, the float value will be multiplied to x before calculate revancy. If is 1.0, means no scale used. Default: "sqr"
        relevancy_method (str): Default: "dot_product"        
    Input:
        x: B, L, C. B: batch size, L: sequence length, C: dimention      
    """
    def __init__(self, in_feature_channels=256, num_intervals=8,  
                xyz_bias=True, end_bias = True, r_scale="sqr", norm_layer=nn.LayerNorm, 
                relevancy_method = "dot_product"):
        super().__init__()             
        self.num_intervals = num_intervals        
        self.norm = norm_layer(in_feature_channels)
        self.xyz_linear = nn.Linear(in_feature_channels, in_feature_channels * 3, bias=xyz_bias)  
        relevancy_size = in_feature_channels // num_intervals
        if isinstance(r_scale, (int, float)):
            self.r_scale_value = r_scale
        elif r_scale == "sqr":            
            self.r_scale_value = 1/ (relevancy_size ** 0.5)
        else:
            self.r_scale_value = 1.0 
        self.r_scale_type = r_scale
        self.softmax = nn.Softmax(dim=-1)        
        self.end_linear = nn.Linear(in_feature_channels, in_feature_channels, bias=end_bias)
        self.relevancy_method = relevancy_method       
        
    def forward(self, input_x):
        """
        Args:
            input features with shape of (B, L, C) B: batch size, L: sequence length, C: dimention
        """        
        b, l, c = input_x.shape
        ch = c//self.num_intervals
        x= self.norm(input_x)
    
        x = self.xyz_linear(x).reshape(b, l, 3, self.num_intervals, ch).permute(2, 0, 3, 1, 4)                
        x, y, z = torch.unbind(x, dim=0) 
        if self.relevancy_method ==  "dot_product":
            if self.r_scale_value != 1.0:                               
                x =  x * self.r_scale_value
            relevancy = torch.matmul(x, y.transpose(-2,-1)) 
        elif self.relevancy_method == "anti_dot_product":
            if self.r_scale_value != 1.0:                
                x =  x * self.r_scale_value
            relevancy = -torch.matmul(x, y.transpose(-2,-1))       

        elif self.relevancy_method == "interval_square_diff":
            mean_factor = self.r_scale_value/ch 
            relevancy = -((((x**2)*mean_factor)[:, :, :, None, :] + ((y**2)*mean_factor)[:,:,None,:,:]).sum(dim=-1) - torch.matmul((2*mean_factor)*x, y.transpose(-1,-2)))            
        elif self.relevancy_method == "anti_interval_square_diff":
            mean_factor = self.r_scale_value/ch
            relevancy = (((x**2)*mean_factor)[:, :, :, None, :] + ((y**2)*mean_factor)[:,:,None,:,:]).sum(dim=-1) - torch.matmul((2*mean_factor)*x, y.transpose(-1,-2)) 
        elif self.relevancy_method == "interval_abs_diff":           
            relevancy = -(torch.abs(x[:, :, :, None, :] - y[:,:,None,:,:]).mean(dim=-1))
            if self.r_scale_value != 1.0:
                relevancy = relevancy * self.r_scale_value            
        elif self.relevancy_method == "anti_interval_abs_diff":         
            relevancy = torch.abs(x[:, :, :, None, :] - y[:,:,None,:,:]).mean(dim=-1)
            if self.r_scale_value != 1.0:
                relevancy = relevancy * self.r_scale_value                  
        else:
            raise ValueError(f'relevancy method {self.relevancy_method} is not found!')

        relevancy = self.softmax(relevancy)                 
        z = torch.matmul(relevancy, z).transpose(-2, -3).contiguous().reshape(b, l, c)    
        z = self.end_linear(z)        
        input_x = input_x + z
        return input_x
    
    
class Block(nn.Module):
    def __init__(
            self,
            in_feature_channels: int,
            num_heads: int,
            mlp_hidden_channel_ratio: float = 4.,            
            qkv_bias: bool = False,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,            
            relev_end_bias = True,
            relev_scale = "sqr",            
            relevancy_method = "dot_product",
        ) -> None:
        super().__init__()
        self.attn =  RelevancyWeightedSum1D(
                        in_feature_channels=in_feature_channels,                        
                        num_intervals=num_heads,
                        xyz_bias=qkv_bias, 
                        end_bias = relev_end_bias, 
                        r_scale=relev_scale, 
                        norm_layer=norm_layer,                         
                        relevancy_method = relevancy_method,                         
                        ) 
        
        self.mlp = Mlp(in_feature_channels = in_feature_channels, 
                hidden_feature_channels=int(in_feature_channels * mlp_hidden_channel_ratio), 
                out_feature_channels=None, 
                act_layer=act_layer,
                norm_layer=norm_layer, 
                )        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x


class PatchEmbed(nn.Module):   
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768, 
        ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = (img_size, img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]           
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)  
    def forward(self, x:torch.Tensor):         
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)       
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                embed_dim=1024, num_classes = 1000, depth=24, num_heads=16,
                relev_scale = "sqr",                
                relevancy_method = "dot_product",
                relev_position_encoding = "sin_cos",                               
                mlp_hidden_channel_ratio=4.0,                 
                norm_layer=nn.LayerNorm, 
                act_layer: nn.Module = nn.GELU,   
                ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches 
        self.relev_position_encoding = relev_position_encoding
        if relev_position_encoding == "sin_cos":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(in_feature_channels=embed_dim, 
                    num_heads=num_heads, 
                    mlp_hidden_channel_ratio=mlp_hidden_channel_ratio,                     
                    qkv_bias=True, 
                    relev_scale=relev_scale,
                    relevancy_method = relevancy_method,
                    norm_layer = norm_layer,                                        
                    act_layer=act_layer
                )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)       
        self.head = nn.Linear(embed_dim, num_classes)           

    def forward_encoder(self, x:torch.Tensor):       
        x = self.patch_embed(x)        
        if self.relev_position_encoding == "sin_cos":
            x = x + self.pos_embed        
        for blk in self.blocks:
            x = blk(x) 

        x = x.mean(dim=1)    
        x = self.norm(x)
        x = self.head(x)
        return x   

    def forward(self, imgs):
        pred  = self.forward_encoder(imgs)
        return pred