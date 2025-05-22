import random
import argparse
import wandb
import numpy as np
from tqdm import tqdm
from datetime import datetime
from datasets.augmix_ops import augmentations
from sklearn.neighbors import LocalOutlierFactor
from Visualizer.visualizer.visualizer import get_local
get_local.activate()

import torch
import torch.nn.functional as F
import operator
import matplotlib.pyplot as plt

import clip
from utils import *

def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of GSBias on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--bs', dest='bs', type=int, required=True, help="number of augmentation views")
    parser.add_argument('--selection_p', dest='selection_p', type=float, required=True, help="selection prob")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./DATA/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args

# Ours
def GSBias(cfg, loader, clip_model, clip_weights, dataset_name, args):
    accuracies = [] 
    hyperparams = {k: cfg[k] for k in ['alpha', 'beta', 'K', 'T']}
    #Test-time adaptation
    for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')): 
        augmented_images = torch.cat(images, dim = 0).cuda()
        images, target = augmented_images[0], target.cuda()
        
        with torch.no_grad():
            global_feature, spatial_faeture = clip_model.encode_image(augmented_images)
        
        global_feature /= global_feature.norm(dim=-1, keepdim=True) # [1, 512]
        spatial_faeture /= spatial_faeture.norm(dim=-1, keepdim=True)
        spatial_faeture = spatial_faeture[0] # [196, 512]
        ori_feature = global_feature[0].unsqueeze(0) # [1, 512]
        
        # B_G
        bias_prompt_g = torch.nn.Parameter(torch.zeros(1, clip_weights.size(1))).cuda() 
        # B_S
        bias_prompt_s = torch.nn.Parameter(torch.zeros(1, clip_weights.size(1))).cuda()           
        loss_history = []
        prompt_history_g = bias_prompt_g
        prompt_history_s = bias_prompt_s

        r_g = 0.9
        r_s = 0.9

        for _ in range(hyperparams['T']):
            prompt_cur_g = r_g * bias_prompt_g + (1 - r_g) * prompt_history_g
            prompt_cur_s = r_s * bias_prompt_s + (1 - r_s) * prompt_history_s
            
            logits = 100. * global_feature.half() @ clip_weights.half()
            logits += prompt_cur_g
            logits, _ = select_confident_samples(logits, args.selection_p)
            
            spatial_logits = 100. * spatial_faeture.half() @ clip_weights.half()
            spatial_logits += prompt_cur_s 
                        
            soft_spatial_logits = F.softmax(spatial_logits, dim=0)
            s_t_score = torch.sum(soft_spatial_logits, dim=1 ,keepdim=True).t() # [196, 1]
            
            spatial_index = torch.topk(s_t_score, k = hyperparams['K'], dim=1)[1] # [1, topk]
            spatial_index = spatial_index.squeeze(0).unsqueeze(1)
            spatial_index = spatial_index.expand(-1, spatial_logits.size(1))
            select_spatial_logits = torch.gather(spatial_logits, dim=0, index=spatial_index)

            loss_g = avg_entropy(logits)
            loss_s = avg_entropy(select_spatial_logits)
            
            loss_history.append(loss_g.item())
            
            if len(loss_history) >= 2:
                loss_change_percent = abs((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100
                if loss_change_percent > 25:
                    break
            
            prompt_history_g = prompt_cur_g
            prompt_history_s = prompt_cur_s
            
            grad_cond_g = torch.autograd.grad(loss_g.requires_grad_(True), [bias_prompt_g], retain_graph=True)[0]
            bias_prompt_g = bias_prompt_g - hyperparams['alpha'] * grad_cond_g
            grad_cond_s = torch.autograd.grad(loss_s.requires_grad_(True), [bias_prompt_s], retain_graph=True)[0]
            bias_prompt_s = bias_prompt_s - hyperparams['beta'] * grad_cond_s
            get_local.clear()
            torch.cuda.empty_cache()

        with torch.no_grad():
            get_local.clear()
            torch.cuda.empty_cache()
            prompt_history_g = r_g * bias_prompt_g + (1 - r_g) * prompt_history_g
            prompt_history_s = r_s * bias_prompt_s + (1 - r_s) * prompt_history_s
            
            logits = 100. * ori_feature.half() @ clip_weights.half()
            logits += prompt_history_g
            logits += prompt_history_s
            get_local.clear()
            torch.cuda.empty_cache()

        acc = cls_acc(logits, target)  
        accuracies.append(acc)
        if i%1000==0:
            print("---- GS-Bias's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
            
    print("---- GS-Bias's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
    with open('outputs/result.txt', 'a') as f:
        f.write("GS-Bias's performance on {}: Top1- {:.2f}\n".format(dataset_name, sum(accuracies)/len(accuracies)))
    
    return sum(accuracies)/len(accuracies)

def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess, args)
        clip_weights = clip_classifier(classnames, template, clip_model) # reuse text classfier

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)

        acc = GSBias(cfg, test_loader, clip_model, clip_weights, dataset_name, args)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()