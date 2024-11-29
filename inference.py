
from dataclasses import asdict
from os import path
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from config import get_model_args, ValArgs
from models import VisionTransformer
import utils

@torch.compile(disable=ValArgs.disable_compile, fullgraph=False)
def inference_one_iter(model, samples, enable_amp=True):    
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=enable_amp):
        pred = model(samples)
    return pred

@torch.no_grad()
def evaluate(model:torch.nn.Module, 
             data_loader:torch.utils.data.DataLoader, 
             topk = (1, 5), 
             num_classes = 1000):
    model.eval()
    device = next(model.parameters()).device    
    accuracy = utils.accuracy(topk= topk, device=device, num_classes=num_classes)
    for data_iter_step, (samples, labels)in enumerate(data_loader, 0):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = inference_one_iter(model, samples) 
        accuracy.step_accuray(pred, labels)   
    
    topk_acc, top1_acc_per_class = accuracy.get_final_accuracy() 
    return topk_acc, top1_acc_per_class


def main():
    val_args = ValArgs()
    dataset_num_classes = {"CIFAR10":10, 'ImageNet1k':1000}
    model_args = get_model_args(val_args.model_name)
    model_args.num_classes = dataset_num_classes[val_args.dataset]
    device = torch.device(val_args.device)
    cudnn.benchmark = True
    if model_args.img_size <= 224:
        resize_factor = 224/256
    else:
        resize_factor = 1.0
    resize_size = int(model_args.img_size/resize_factor)
    transform_val = transforms.Compose([
                transforms.Resize(resize_size, interpolation=3),
                transforms.CenterCrop(model_args.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    if val_args.dataset == "CIFAR10":    
        dataset_val = datasets.CIFAR10(root=path.join(val_args.dataset_dir, val_args.dataset), train=False, transform=transform_val, download=True) 
    elif 'ImageNet1k' in val_args.dataset:    
        dataset_val = datasets.ImageFolder(path.join(val_args.dataset_dir, val_args.dataset, 'val'), transform=transform_val)


    data_loader_val = torch.utils.data.DataLoader(
            dataset_val,                                        
            batch_size=val_args.batch_size,
            num_workers=val_args.num_workers,
            pin_memory=val_args.pin_memory,
            drop_last=False,
            )

    model = VisionTransformer(
            img_size=model_args.img_size,
            patch_size=model_args.patch_size,
            in_chans=model_args.in_chans,
            embed_dim=model_args.dim,
            num_classes= model_args.num_classes,
            depth=model_args.n_layers,
            num_heads=model_args.n_heads,
            relev_scale=model_args.relev_scale,
            relevancy_method=model_args.relevancy_method,
            relev_position_encoding=model_args.relev_position_encoding,                                                            
            mlp_hidden_channel_ratio=model_args.mlp_hidden_channel_ratio,            
            norm_layer=model_args.norm_layer,                                                         
            )  
    checkpoint = torch.load(val_args.checkpoint_path, map_location='cpu', weights_only=True)
    keys_message = model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    curr_accuracy, curr_top1_acc_per_class = evaluate(model=model, data_loader=data_loader_val, 
                    num_classes=model_args.num_classes, 
                    )
    print(f'final validation accuracy: {curr_accuracy}')


if __name__ == '__main__':    
    main()