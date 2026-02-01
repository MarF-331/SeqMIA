import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..Models import JHUData, JHUDataForDistill, P2PNeXt, P2PNeXtStandardArgs
from .JHU_utils import JHU_DATA_TRANSFORM, jhu_collate_fn

def extractSoftLabelsP2PNext(model: torch.nn.Module, distillation_dataloader: DataLoader, device=torch.device("cpu")):
    model.eval()
    model.to(device)
    soft_labels: list[dict[str, torch.Tensor]] = []

    with torch.no_grad():
        pbar = tqdm(distillation_dataloader, desc="Extracting Soft Labels", leave=False)
        for sample, _ in pbar:
            sample = sample.to(device)
            output = model(sample)
            
            for i in range(distillation_dataloader.batch_size):
                soft_labels.append({
                    "pred_points": output["pred_points"][i].cpu(),
                    "pred_logits": output["pred_logits"][i].cpu()
                })

    return soft_labels


def getDistillationDataLoaderP2PNext(teacher: nn.Module, distill_image_data: list[tuple[str, np.ndarray]], 
                                     batch_size: int=1, num_workers: int=4, 
                                     device=torch.device("cpu")) -> DataLoader:
    
    distill_dataset = JHUData(distill_image_data, JHU_DATA_TRANSFORM, center_crop=512)
    distill_dataloader = DataLoader(distill_dataset, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers, 
                                    shuffle=False, 
                                    collate_fn=jhu_collate_fn)                                              
    
    soft_labels = extractSoftLabelsP2PNext(teacher, distill_dataloader, device)
    
    distill_data_with_soft_labels = JHUDataForDistill(distill_dataset, soft_labels)
    distill_loader_with_soft_labels = DataLoader(distill_data_with_soft_labels,
                                                 batch_size=batch_size, 
                                                 shuffle=True, 
                                                 num_workers=num_workers, 
                                                 collate_fn=jhu_collate_fn)
    
    return distill_loader_with_soft_labels


def load_distillation_models(distill_models_dir: str, model_args):
    if not os.path.exists(distill_models_dir):
        raise FileNotFoundError(f"Distillation Models Directory not found: {distill_models_dir}")
    else:
        distill_file_names = sorted([f for f in os.listdir(distill_models_dir) if f.endswith(".pth")])
        if len(distill_file_names) == 0:
            raise ValueError(f"No .pth files found in directory {distill_models_dir}")
        else:
            distill_paths = [os.path.join(distill_models_dir, f) for f in distill_file_names]
            distill_models = [P2PNeXt(model_args, checkpoint_path=p) for p in distill_paths]
            distill_model_ids = [m.epoch if not m.epoch is None else m.checkpoint_path for m in distill_models]
            distill_model_and_ids = list(zip(distill_models, distill_model_ids))

            return distill_model_and_ids