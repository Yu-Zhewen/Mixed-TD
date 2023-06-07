import copy
import os
import thop
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets

from models.repvgg import get_RepVGG_func_by_name, repvgg_model_convert
from thop.vision.basic_hooks import zero_ops
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import transforms

def replace_modules(model, replace_dict):
    for name, module in model.named_modules(): 
        for subname, submodule in module.named_children():
            if submodule in replace_dict.keys():
                new_submodule = replace_dict[submodule]
                assert(hasattr(module, subname))
                setattr(module,subname,new_submodule)

class BasicBlockReluFixed(nn.Module):
    def __init__(self, basic_block):
        super(BasicBlockReluFixed, self).__init__()
        assert isinstance(basic_block, BasicBlock)
        self.conv1 = copy.deepcopy(basic_block.conv1)
        self.bn1 = copy.deepcopy(basic_block.bn1)
        self.relu1 = nn.ReLU()
        self.conv2 = copy.deepcopy(basic_block.conv2)
        self.bn2 = copy.deepcopy(basic_block.bn2)
        self.relu2 = nn.ReLU()
        self.downsample = basic_block.downsample
        self.stride = basic_block.stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

def fix_resnet_relu(model):
    replace_dict = {}
    for name, module in model.named_modules(): 
        if isinstance(module, BasicBlock):
            replace_dict[module] = BasicBlockReluFixed(module)
    replace_modules(model, replace_dict)
    return model

def load_model(model_name, pretrained=True):
    if model_name == "repvgga0":
        model = get_RepVGG_func_by_name('RepVGG-A0')()
        model.load_state_dict(torch.load('models/RepVGG-A0-train.pth'))
        model = repvgg_model_convert(model)
    elif model_name == "resnet18":
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        model = fix_resnet_relu(model)

    input_size = (1, 3, 224, 224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
    train_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

    model.input_size = input_size
    model.train_transforms = train_transforms
    model.val_transforms = val_transforms

    return model

def annoate_feature_map_size(model, input):
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        input = input.cuda()

    def _fhook(module, input, output):
        assert len(input) == 1
        module.ifm_size = input[0].shape
        module.ofm_size = output.shape
    
    handlers = [] 
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            h = module.register_forward_hook(_fhook)
            handlers.append(h)
    model(input)    
    for h in handlers:
        h.remove()

def prepare_dataloader(model, batch_size, workers, train_split_size=0, batch_size_multi=1, drop_last=False):
    data = os.path.expanduser("~/dataset/ILSVRC2012_img")

    valdir = os.path.join(data, 'val')
    traindir = os.path.join(data, 'train')

    val_dataset = datasets.ImageFolder(valdir, model.val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(batch_size_multi*batch_size), shuffle=False,
        num_workers=workers, pin_memory=True)

    train_dataset = datasets.ImageFolder(traindir, model.train_transforms)
    assert train_split_size % 1000 == 0
    rand_indexes = torch.randperm(len(train_dataset)).tolist()
    train_labels = [sample[1] for sample in train_dataset.samples]
    per_class_remain = [train_split_size // 1000] * 1000
    train_indexes, sub_indexes = [], []
    for idx in rand_indexes:
        label = train_labels[idx]
        if per_class_remain[label] > 0:
            sub_indexes.append(idx)
            per_class_remain[label] -= 1
        else:
            train_indexes.append(idx)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(sub_indexes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,  
        batch_size=batch_size,
        num_workers=workers, pin_memory=True, sampler=train_sampler, 
        drop_last=drop_last)

    sub_dataset = datasets.ImageFolder(traindir, model.val_transforms)
    sub_loader = torch.utils.data.DataLoader(
        sub_dataset, 
        batch_size=batch_size,
        num_workers=workers, pin_memory=True, sampler=sub_sampler)

    return train_loader, sub_loader, val_loader

def calculate_macs_params(model, input, turn_on_warnings=False, verbose=True):
    if torch.cuda.is_available():
        model.cuda()
        input = input.cuda()
    model.eval()
    if isinstance(model, torch.nn.DataParallel):
        model_copy = copy.deepcopy(model.module)
    else:
        model_copy = copy.deepcopy(model)

    macs, params = thop.profile(model_copy, inputs=(input, ), custom_ops={nn.BatchNorm2d:zero_ops}, verbose=turn_on_warnings)
    format_macs, format_params = thop.clever_format([macs, params], "%.3f")
    if verbose:
        print("MACs:", format_macs, "Params:", format_params)
    return macs, params

def train(train_loader, model, criterion, optimizer, epoch, print_freq=0, 
    scheduler=None, model_ema = None, scaler=None, clip_grad_norm=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        if model_ema and i % model_ema.steps == 0:
            model_ema.update_parameters(model)
            if epoch < model_ema.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if print_freq != 0 and i % print_freq == 0:
            progress.display(i)

        
    print(' * Train * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1, top5

def validate(val_loader, model, criterion, print_freq=0, verbose=True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if print_freq != 0 and i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        if verbose:
            print(' * Valid * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1, top5


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,), no_reduce=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        if no_reduce:
            return correct

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
