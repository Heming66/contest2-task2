import argparse
import os
import time

import torch
import torchmetrics
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from datasets import MMACDataset, get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils import AUC, Specificity,Dice,set_seed,get_model


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        if type(prediction) == type(()):
            prediction = prediction[0]
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            if type(prediction) == type(()):
                prediction = prediction[0]
            loss = self.loss(prediction, y)
        return loss, prediction



def get_optimizer(name,lr,params):
    if name == 'adam':
        return torch.optim.Adam([
            dict(params=params, lr=lr),
        ])
    elif name == 'adamw':
        return torch.optim.AdamW(
            params=params, lr=lr, weight_decay=1e-4
            )
    elif name == 'asgd':
        return torch.optim.ASGD(
            params=params, lr=lr
            )
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_scheduler', type=str, default='step')
    parser.add_argument('--model', type=str, default='unet++')
    parser.add_argument('--opti', type=str, default='adam')
    parser.add_argument('--lesion', type=str, default='LC')


    #model_set
    parser.add_argument('--encoder', type=str, default='resnext50_32x4d')
    parser.add_argument('--encoder_weights', type=str, default='imagenet')
    parser.add_argument('--activation', type=str, default='sigmoid')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    set_seed(args.seed)

    # create segmentation model with pretrained encoder
    model = get_model(args)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
   
    train_dataset = MMACDataset(
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        lesion=args.lesion,
    )

    valid_dataset = MMACDataset(
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        lesion=args.lesion,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        Dice(threshold=0.5)
    ]

    optimizer = get_optimizer(args.opti,args.lr,model.parameters())

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.6)
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=args.device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=args.device,
        verbose=True,
    )
    if not os.path.exists('./models/'+args.lesion):
        os.makedirs('./models/'+args.lesion)

    if not os.path.exists('./logs/'+args.lesion):
        os.makedirs('./logs/'+args.lesion)

    current_time = str(int(round(time.time() * 1000)))

    log_txt = open('./logs/'+args.lesion+'/'+current_time+'.txt', 'w')
    
    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    log_txt.write(message)
    log_txt.flush()

    max_score = 0
    best_epoch = 0

    for i in range(0, args.epochs):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)

        valid_logs = valid_epoch.run(valid_loader)

        log_txt.write('\nEpoch: {} '.format(i))
        log_txt.write('iou_score: {} '.format(valid_logs['iou_score']))
        log_txt.write('dice_score: {} '.format(valid_logs['dice']))
        log_txt.flush()

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['dice']:
            max_score = valid_logs['dice']
            best_epoch = i
            torch.save(model, './models/'+args.lesion+'/'+current_time+'.pth')
            print('Model saved!')
        
        scheduler.step()


    log_txt.write('\nBest epoch: {} '.format(best_epoch))
    log_txt.write('Best dice_score: {} '.format(max_score))
    log_txt.flush()
    log_txt.close()
