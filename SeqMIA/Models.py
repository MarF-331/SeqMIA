from re import X
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Callable
from PIL import Image
import numpy as np
import torch.nn.functional as F
import math
import os
import sys
import torch.nn.utils.rnn as rnn_utils
import random
from .utils.crowd_data_utils import resize_image_to_target_size, resize_to_multiple_of_128,\
    center_crop_image, random_crop_image

current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_p2p_next = os.path.join(current_dir, "../P2PNeXt")
path_to_dm_count = os.path.join(current_dir, "../DMCount")
sys.path.append(path_to_p2p_next)
sys.path.append(path_to_dm_count)

from P2PNeXt.Networks.P2P.models.p2pnet_conv_next import build
from P2PNeXt.utils import run_P2P_inference

from DMCount.models import vgg19
from DMCount.datasets.crowd import gen_discrete_map

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FcBlock(nn.Module):
    def __init__(self, fc_params, flatten):
        super(FcBlock, self).__init__()
        input_size = int(fc_params[0])
        output_size = int(fc_params[1])

        fc_layers = []
        if flatten:
            fc_layers.append(Flatten())
        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd


class ConvBlock(nn.Module):
    def __init__(self, conv_params):
        super(ConvBlock, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        avg_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        conv_layers = []
        conv_layers.append(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))

        conv_layers.append(nn.ReLU())

        if avg_pool_size > 1:
            conv_layers.append(nn.AvgPool2d(kernel_size=avg_pool_size))

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd


class VGG(nn.Module):
    def __init__(self, params):
        super(VGG, self).__init__()

        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.conv_channels = params['conv_channels']
        self.fc_layer_sizes = params['fc_layers']

        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        self.init_weights = params['init_weights']
        self.augment_training = params['augment_training']
        self.num_output = 1

        self.init_conv = nn.Sequential()

        self.layers = nn.ModuleList()
        input_channel = 3
        cur_input_size = self.input_size
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size / 2)
            conv_params = (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            self.layers.append(ConvBlock(conv_params))
            input_channel = channel

        fc_input_size = cur_input_size * cur_input_size * self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width

        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        fwd = self.init_conv(x)

        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class TrData(Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]


def collate_fn(trs):
    onetr = trs[0]
    onepoint_size = onetr.size(1)
    input_size = onepoint_size - 1
    trs.sort(key=lambda x: len(x), reverse=True)
    tr_lengths = [len(sq) for sq in trs]
    trs = rnn_utils.pad_sequence(trs, batch_first=True, padding_value=0)
    var_x = trs[:, :, 1:input_size + 1]
    tmpy = trs[:, :, 0]
    var_y = tmpy[:, 0]
    return var_x, var_y, tr_lengths


class lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, num_layers=1, num_classes=2, batch_size=1):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, len_of_oneTr):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        batch_x_pack = rnn_utils.pack_padded_sequence(x,
                                                      len_of_oneTr, batch_first=True).cuda()
        out, (h1, c1) = self.layer1(batch_x_pack, (h0, c0))
        out = self.layer2(h1)
        return out, h1


class LSTM_Attention(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, num_layers=1, num_classes=2, batch_size=1):
        super(LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer3 = nn.Linear(hidden_size * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, len_of_oneTr):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        batch_x_pack = rnn_utils.pack_padded_sequence(x,
                                                      len_of_oneTr, batch_first=True).cuda()
        out, (h1, c1) = self.layer1(batch_x_pack, (h0, c0))
        outputs, lengths = rnn_utils.pad_packed_sequence(out, batch_first=True)
        permute_outputs = outputs.permute(1, 0, 2)
        atten_energies = torch.sum(h1 * permute_outputs,
                                   dim=2)

        atten_energies = atten_energies.t()

        scores = F.softmax(atten_energies, dim=1)

        scores = scores.unsqueeze(0)

        permute_permute_outputs = permute_outputs.permute(2, 1, 0)
        context_vector = torch.sum(scores * permute_permute_outputs,
                                   dim=2)
        context_vector = context_vector.t()
        context_vector = context_vector.unsqueeze(0)
        out2 = torch.cat((h1, context_vector), 2)
        out = self.layer3(out2)
        return out, out2


@dataclass
class P2PNeXtStandardArgs():
    backbone: str = "conv_next"
    backbone_type: str = "conv_next_tiny"
    feature_map: str = "feature_map 1"
    feature_pyramid: str = "new"
    row: int = 2
    line: int = 2
    point_loss_coef: float = 0.0002
    eos_coef: float = 0.5
    set_cost_class: float = 1
    set_cost_point: float = 0.05


class P2PNeXt(nn.Module):
    '''
    A wrapper for the P2PNeXt model.
    This wrapper holds the model itself and the criterion.
    '''
    def __init__(self, args, checkpoint_path=None):
        super(P2PNeXt, self).__init__()
        self.args = args
        self.model, self.criterion = self._build_model()
        self.epoch: int | None = None
        self.checkpoint_path: str | None = checkpoint_path
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
    def _build_model(self):
        model, criterion = build(args=self.args, training=True)
        return model, criterion
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint at: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            if not checkpoint.get("epoch") is None:
                self.epoch = checkpoint["epoch"]
            self.checkpoint_path = checkpoint_path
            print(f"Checkpoint loaded!")
        else:
            print(f"Path was not found: {checkpoint_path}")

    def run_inference(self, image_path_list: list[str], image_transformation: str=None, save_directory: str=None, save_file_type: str=None, device: torch.device=torch.device("cpu")):
        predictions = run_P2P_inference(image_path_list, self.model, image_transformation, 0.5, save_directory, save_file_type, device)
        return predictions

    def forward(self, x):
        return self.model(x)


class DMCount(nn.Module):
    def __init__(self, checkpoint_path: str=None):
        super(DMCount, self).__init__()
        self.model = vgg19()
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
    
    def forward(self, x):
        return self.model(x)
    
    def predict_count(self, x):
        output, _ = self.forward(x)
        count = [int(output[i].sum().item()) for i in range(x.shape[0])]
        return count

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if os.path.exists(checkpoint_path):
            print(f"Loading DMCount Checkpoint at {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint)
            print(f"DMCount Checkpoint loaded!")
        else:
            print(f"Path was not found {checkpoint_path}")


class AttackRNNP2PNeXt(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int=1):
        super(AttackRNNP2PNeXt, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        out = self.fc(hn[-1])
        return out
    

class CIFARData(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        img = self.X_train[idx]
        label = self.y_train[idx]
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label


class CIFARDataForDistill(Dataset):
    def __init__(self, X_train, y_train, softlabel):
        self.X_train = X_train
        self.y_train = y_train
        self.softlabel = softlabel

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        img = self.X_train[idx]
        label = self.y_train[idx]
        softlabel = self.softlabel[idx]
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)
        return img, label, softlabel


class CrowdData(Dataset):
    def __init__(self, image_data: list[tuple[str, np.ndarray]], 
                 transform :Callable[[Image.Image], Image.Image]=None, 
                 fixed_image_size: tuple[int, int]=None, random_crop: int=None,
                 center_crop: int=None, resize_to_multiple_of_128: bool=True,
                 generate_discrete_map: bool=False, discrete_map_downsample: int=8):
        
        self.image_data = sorted(image_data, key=lambda tup: os.path.basename(tup[0]))
        self.transform = transform
        self.fixed_image_size = fixed_image_size
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.resize_to_multiple_of_128 = resize_to_multiple_of_128
        self.generate_discrete_map = generate_discrete_map
        self.downsample_factor = discrete_map_downsample
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path, ground_truth_points = self.image_data[index]
        img_raw = Image.open(image_path).convert("RGB")
       
        if self.fixed_image_size:
            img_raw, ground_truth_points = resize_image_to_target_size(img_raw, ground_truth_points, self.fixed_image_size)
        
        if self.random_crop:
            img_raw, ground_truth_points = random_crop_image(img_raw, ground_truth_points, self.random_crop)

        elif self.center_crop:
            img_raw, ground_truth_points = center_crop_image(img_raw, ground_truth_points, self.center_crop)

        if self.resize_to_multiple_of_128:
            img_raw, ground_truth_points = resize_to_multiple_of_128(img_raw, ground_truth_points)
        
        if self.generate_discrete_map:
            width, height = img_raw.size
            discrete_map = gen_discrete_map(height, width, ground_truth_points)
            discrete_map = discrete_map.reshape(height // self.downsample_factor, self.downsample_factor,
                                                width // self.downsample_factor, self.downsample_factor).sum(axis=(1, 3))
            discrete_map = np.expand_dims(discrete_map, 0)
        
        if self.transform:
            img_tensor = self.transform(img_raw)
        else:
            to_tensor = transforms.Compose([transforms.ToTensor()])
            img_tensor = to_tensor(img_raw)
        
        img_id = int(os.path.basename(image_path).split(".")[0])

        target = {
            "point": torch.tensor(ground_truth_points, dtype=torch.float32),
            "image_id": torch.tensor([img_id], dtype=torch.long),
            "labels": torch.ones(ground_truth_points.shape[0], dtype=torch.long)
        }

        if self.generate_discrete_map:
            target["discrete_map"] = torch.tensor(discrete_map, dtype=torch.float32)

        return img_tensor, target


class CrowdDataForDistill(Dataset):
    def __init__(self, image_data_set: CrowdData, soft_labels: list[dict[str, torch.Tensor]]):
        self.data = image_data_set
        self.soft_labels = soft_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        img, _ = self.data[index]
        soft_label = self.soft_labels[index]
        
        return img, soft_label
    

class P2PNeXtDistillationLoss(nn.Module):
    def __init__(self, point_loss_weight: float=1.0):
        super().__init__()
        self.point_loss_weight = point_loss_weight
    
    def forward(self, student_output_raw: dict[str, torch.Tensor], soft_labels: list[dict[str, torch.Tensor]]):
        '''
        Compute the distillation loss between the student model's output and the soft labels.

        Args:
            student_output_raw (dict[str, torch.Tensor]): The raw output from the student model. Each tensor is of shape [batch_size, num_queries, 2].
            soft_labels (list[dict[str, torch.Tensor]]): The soft labels obtained from the teacher model. Each index corresponds to a batch item and contains
                - 'pred_points': Tensor of shape [num_queries, 2] with the predicted point coordinates from the teacher model.
                - 'pred_logits': Tensor of shape [num_queries, 2] with the logits from the teacher model.
        
        Returns:
            torch.Tensor: The computed distillation loss.
        '''
        batch_size = student_output_raw['pred_points'].shape[0]
        device = student_output_raw['pred_points'].device
        total_loss = 0.0

        for i in range(batch_size):
            student_points = student_output_raw['pred_points'][i]  # [num_queries, 2]
            student_logits = student_output_raw['pred_logits'][i]  # [num_queries, 2]
            teacher_points = soft_labels[i]['pred_points'].to(device)  # [num_queries, 2]
            teacher_logits = soft_labels[i]['pred_logits'].to(device)  # [num_queries, 2]

            # Compute confidence loss (KL divergence)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            confidence_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

            # Compute point loss (weighted L2 loss)
            teacher_confidences = teacher_probs[:, 1].unsqueeze(-1)  # Confidence for the positive class
            point_loss = (teacher_confidences * (student_points - teacher_points) ** 2).mean()

            total_loss += self.point_loss_weight * point_loss + confidence_loss

            batch_loss = total_loss / batch_size

        return batch_loss
