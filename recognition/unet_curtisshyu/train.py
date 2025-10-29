from .modules import UNet
from .utils import param_check

model = UNet(n_channels=1, n_classes=1)
_, params = param_check(model)
