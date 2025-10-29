from recognition.unet_curtisshyu.modules import UNet
from recognition.unet_curtisshyu.utils import param_check

model = UNet(n_channels=1, n_classes=1)
_, params = param_check(model)
