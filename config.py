from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class ModelArgsCentiBase:
    dim: int = 320
    n_layers: int = 8
    n_heads: int = 10    
    img_size:int  = 224
    patch_size: int = 16
    in_chans: int = 3
    relev_scale: str = "sqr" 
    relevancy_method: str = "dot_product" 
    relev_position_encoding: str = "sin_cos"
    mlp_hidden_channel_ratio: float = 4.0 
    norm_layer: nn.Module = nn.LayerNorm
    act_layer: nn.Module = nn.GELU    
    num_classes: int = 1000    
    

def get_model_args(model_name):
    """
    get dataclass for model args under specified model name.    
    """
    
    if model_name == 'centi_anti_dot':
        @dataclass
        class ModelArgs(ModelArgsCentiBase):
            relevancy_method: str = "anti_dot_product"            
        return ModelArgs()
    if model_name == 'centi_dot':
        @dataclass
        class ModelArgs(ModelArgsCentiBase):
            relevancy_method: str = "dot_product"            
        return ModelArgs()   
    
    raise ValueError(f'the model name: {model_name} is not defined.')
    
@dataclass
class ValArgs:  
    model_name: str = 'centi_dot' 
    dataset: str = 'ImageNet1k'
    checkpoint_path: str = ''#'weights/centi_dot.pth'
    batch_size: int = 256 
    disable_compile: bool = True 
    num_classes: int = 1000
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"        
    num_workers: int = 8
    pin_memory: bool = True 
    dataset_dir: str = ''#'image_data'