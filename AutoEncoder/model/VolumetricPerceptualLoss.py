import torch
import torch.nn as nn
import os 
def volume_standardize(input):
    input = input.float()
    mean = input.mean()
    std = input.std(unbiased=input.numel() > 1).clamp_min(1e-6)
    return (input - mean) / std
def feature_l2_normalize(x , eps  = 1e-7):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)
def spatial_average(x, keepdim = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)
class VolumetricPerceptualLoss(nn.Module):
    def __init__(
        self, net_path = None, verbose = True, channel_wise = False):
        super().__init__()
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        net = "volumetric_resnet10_23datasets"
        backbone_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../perceptual_backbone'
        if net_path is None:
            net_path = backbone_dir
        self.model = torch.hub.load(repo_or_dir=net_path, model=net, trust_repo=True, source='local',model_dir = backbone_dir + '/weights')
        self.eval()

        self.channel_wise = channel_wise

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        input = volume_standardize(input)
        target = volume_standardize(target)

        # Get model outputs
        feats_per_ch = 0
        for ch_idx in range(input.shape[1]):
            input_channel = input[:, ch_idx, ...].unsqueeze(1)
            target_channel = target[:, ch_idx, ...].unsqueeze(1)

            if ch_idx == 0:
                outs_input = self.model.forward(input_channel)
                outs_target = self.model.forward(target_channel)
                feats_per_ch = outs_input.shape[1]
            else:
                outs_input = torch.cat([outs_input, self.model.forward(input_channel)], dim=1)
                outs_target = torch.cat([outs_target, self.model.forward(target_channel)], dim=1)


        feats_input = feature_l2_normalize(outs_input)
        feats_target = feature_l2_normalize(outs_target)

        feats_diff: torch.Tensor = (feats_input - feats_target) ** 2
        if self.channel_wise:
            results = feats_diff.new_zeros(
                feats_diff.shape[0], input.shape[1], feats_diff.shape[2], feats_diff.shape[3], feats_diff.shape[4]
            )
            for i in range(input.shape[1]):
                l_idx = i * feats_per_ch
                r_idx = (i + 1) * feats_per_ch
                results[:, i, ...] = feats_diff[:, l_idx:r_idx, ...].sum(dim=1)
        else:
            results = feats_diff.sum(dim=1, keepdim=True)

        results = results.mean([2, 3, 4], keepdim=True)

        return results
