"""
Reward model wrappers for distillation experiments.
"""
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T


class ClassifierReward:
    """
    Use pretrained ResNet-50 confidence as reward:
      reward = p(class = c | image)
    """

    def __init__(self, device):
        self.device = device
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        self.model = tv_models.resnet50(weights=weights).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    @torch.no_grad()
    def __call__(self, images_pixel, class_labels):
        """
        Args:
            images_pixel: (B, 3, H, W), value range [-1, 1]
            class_labels: (B,)
        Returns:
            rewards: (B,), in [0, 1]
        """
        # Keep reward model in fp32 for numerical stability and dtype consistency.
        x = (images_pixel * 0.5 + 0.5).clamp(0, 1).float()
        x = self.normalize(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        labels = class_labels.to(device=probs.device, dtype=torch.long)
        rewards = probs[torch.arange(probs.size(0), device=probs.device), labels]
        return rewards


class LPIPSReward:
    """
    Optional LPIPS-based reward:
      reward = -LPIPS(x_exp, x_hq)
    """

    def __init__(self, device):
        import lpips

        self.fn = lpips.LPIPS(net="vgg").eval().to(device)
        for p in self.fn.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, x1_exp_pixel, x1_hq_pixel):
        dist = self.fn(x1_exp_pixel, x1_hq_pixel).squeeze()
        return -dist
