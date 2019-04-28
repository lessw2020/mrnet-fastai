import numpy as np
from fastai.vision import *
import torch

data_path = Path('../data')
sag_path = data_path/'sagittal'
cor_path = data_path/'coronal'
ax_path = data_path/'axial'

weights = torch.load('loss_weights.pt')

class MRNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.alexnet(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        # in the original code, the input was squeezed here, but this won't work with fastai
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        return torch.sigmoid(self.classifier(x))

    def __call__(self, x): return self.forward(x)


class WtBCELoss(nn.Module):
    def __init__(self, wts):
        super().__init__()
        self.wts = wts.float()

    def forward(self, output, target):
        loss = self.wts[0]*(target.float() * torch.log(output).float()) + self.wts[1]*((1-target).float() * torch.log(1-output).float())
        return torch.neg(torch.mean(loss))


class MR3DImDataBunch(ImageDataBunch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def one_batch(self, ds_type:DatasetType=DatasetType.Train, detach:bool=True, denorm:bool=True, cpu:bool=True)->Collection[Tensor]:
        "Get one batch from the data loader of `ds_type`. Optionally `detach` and `denorm`."
        dl = self.dl(ds_type)
        w = self.num_workers
        self.num_workers = 0
        try:     x,y = next(iter(dl))
        finally: self.num_workers = w
        if detach: x,y = to_detach(x,cpu=cpu),to_detach(y,cpu=cpu)
        norm = getattr(self,'norm',False)
        if denorm and norm:
            x = self.denorm(x)
            if norm.keywords.get('do_y',False): y = self.denorm(y, do_x=True)
        x = torch.squeeze(x, dim=0) # squeeze needed here for learn.summary()
        return x,y


class MR3DImageList(ImageList):
    _bunch = MR3DImDataBunch # necessary for Data Block API functionality
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_slc = 51 # optimized for sagittal image stacks only...TODO rewrite for any max stack size
        self.c = 1

    # pads on both sides of image stack with zero arrays to equal max_slc
    def open(self, fn):
        x = np.load(fn)
        if x.shape[0] < self.max_slc:
            x_pad = np.zeros((self.max_slc, 256, 256))
            mid = x_pad.shape[0] // 2
            up = x.shape[0] // 2
            if x.shape[0] % 2 == 1: x_pad[mid-up:mid+up+1] = x
            else: x_pad[mid-up:mid+up] = x
        else:
            x_pad = x
        return self.arr2image(np.stack([x_pad]*3, axis=1))

    # converts np.ndarray to fastai Image class
    @staticmethod
    def arr2image(arr:np.ndarray, div:bool=True, cls:type=Image):
        x = Tensor(arr)
        if div == True: x.div_(255)
        return cls(x)


# squeeze input prior to loss calculation
class MRNetCallback(Callback):
    def on_batch_begin(self, last_input, **kwargs):
        x = torch.squeeze(last_input, dim=0)
        return dict(last_input=x)


class MRNetLearner(Learner):
    # redefine specifically for MRNet layer groups
    def freeze_to(self, n:int)->None:
        "Freeze layers up to layer group `n`."
        for g in self.layer_groups[:n]:
            if is_listy(g):
                for l in g:
                    if not self.train_bn or not isinstance(l, bn_types): requires_grad(l, False)
            else: requires_grad(g, False)
        for g in self.layer_groups[n:]: requires_grad(g, True)
        self.create_opt(defaults.lr)

    def freeze(self)->None:
        "Freeze up to the last layer group."
        self.freeze_to(-1)
        self.create_opt(defaults.lr)

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)
        self.create_opt(defaults.lr)


def mrnet_learner(data:DataBunch, model:Callable=MRNet(), pretrained:bool=True, init=nn.init.kaiming_normal_, **kwargs:Any)->Learner:
    _layer_groups = [model.model.features, model.model.avgpool, model.model.classifier, model.gap, model.classifier]
    learn = MRNetLearner(data, model, layer_groups=_layer_groups, **kwargs)
    if pretrained: learn.freeze()
    if init: apply_init(model.classifier, init)
    return learn
