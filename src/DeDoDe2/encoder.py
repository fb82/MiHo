import torch
import torch.nn as nn
import torchvision.models as tvm


class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(weights=pretrained).features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast("cuda", enabled=self.amp, dtype = self.amp_dtype):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes

class VGG(nn.Module):
    def __init__(self, size = "19", pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if size == "11":
            self.layers = nn.ModuleList(tvm.vgg11_bn(weights=pretrained).features[:22])
        elif size == "13": 
            self.layers = nn.ModuleList(tvm.vgg13_bn(weights=pretrained).features[:28])
        elif size == "19": 
            self.layers = nn.ModuleList(tvm.vgg19_bn(weights=pretrained).features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast("cuda", enabled=self.amp, dtype = self.amp_dtype):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes

class FrozenDINOv2(nn.Module):
    def __init__(self, amp = True, amp_dtype = torch.float16, dinov2_weights = None):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        from .transformer import vit_large
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )
        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    def forward(self, x):
        B, C, H, W = x.shape
        if self.dinov2_vitl14[0].device != x.device:
            self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
        with torch.inference_mode():
            dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
            features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
        return [features_16.clone()], [(H//14, W//14)] # clone from inference mode to use in autograd

class VGG_DINOv2(nn.Module):
    def __init__(self, vgg_kwargs = None, dinov2_kwargs = None):
        assert vgg_kwargs is not None and dinov2_kwargs is not None, "Input kwargs pls"
        super().__init__()        
        self.vgg = VGG(**vgg_kwargs)
        self.frozen_dinov2 = FrozenDINOv2(**dinov2_kwargs)
        
    def forward(self, x):
        feats_vgg, sizes_vgg = self.vgg(x)
        feat_dinov2, size_dinov2 = self.frozen_dinov2(x)
        return feats_vgg + feat_dinov2, sizes_vgg + size_dinov2
