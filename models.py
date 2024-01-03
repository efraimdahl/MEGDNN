import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support
import wandb

class MEGNet(pl.LightningModule):
    def __init__(self, config):
        super(MEGNet, self).__init__()
        self.config = config
        self.layers = nn.ModuleList()
        if config['window_size'] != -1:
            input_size = config['window_size']*config['n_sensors']
        else:
            input_size = config['n_timesteps']//config['downsample']*config['n_sensors']
        self.layers.append(nn.Linear(input_size, config['hidden_size']))
        self.layers.append(nn.BatchNorm1d(config['hidden_size']))
        self.layers.append(nn.LeakyReLU(config['negative_slope']))
        self.layers.append(nn.Dropout(config['dropout']))

        for i in range(config['n_layers']-1):
            self.layers.append(nn.Linear(config['hidden_size'], config['hidden_size']))
            self.layers.append(nn.BatchNorm1d(config['hidden_size']))
            self.layers.append(nn.LeakyReLU(config['negative_slope']))
            self.layers.append(nn.Dropout(config['dropout']))
        self.layers.append(nn.Linear(config['hidden_size'], config['num_classes']))

        self.criterion = nn.CrossEntropyLoss()
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.val_progress = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        self.train_loss += loss.item()
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        self.val_loss += loss.item()
        # save predictions and true labels to calculate accuracy on full validation after epoch end
        pred = F.softmax(logits, dim=1)
        self.val_progress.append((pred, y))
        return loss
    
    def on_validation_epoch_end(self):
        # calculate accuracy on full validation set
        preds = torch.cat([pred for pred, y in self.val_progress], dim=0)
        y = torch.cat([y for pred, y in self.val_progress], dim=0)
        preds = torch.argmax(preds, dim=1)
        acc = torch.sum(preds == y).item() / len(y)
        precision, recall, f1, _ = precision_recall_fscore_support(y.cpu(), preds.cpu(), average='macro')
        if self.config['log'] in ['wandb', 'all']:
            wandb.log({'train_loss': self.train_loss, 'val_loss': self.val_loss, 'val_acc': acc, 'val_precision': precision, 'val_recall': recall, 'val_f1': f1})
        if self.config['log'] in ['stdout', 'all']:
            print(f'Training loss: {self.train_loss}')
            print(f'Validation accuracy: {acc}')
            print(f'Validation loss: {self.val_loss}')
            print(f'Validation precision: {precision}')
            print(f'Validation recall: {recall}')
            print(f'Validation f1: {f1}')
        self.val_progress = []
        self.val_loss = 0.0
        self.train_loss = 0.0
    
    def configure_optimizers(self):
        # initialize Adam optimizer and scheduler with warmup
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config['lr'], steps_per_epoch=self.config['steps_per_epoch'], epochs=self.config['epochs'])
        return [self.optimizer], [self.scheduler]


class MEGConvNet(pl.LightningModule):
    def __init__(self, config):
        super(MEGConvNet, self).__init__()
        self.config = config

        # calculate input size for the final fully connected layer
        if config['window_size'] != -1:
            input_size = config['window_size']
        else:
            input_size = config['n_timesteps']//config['downsample']
        
        current_size = input_size
        for _ in range(config['n_layers']):
            current_size = ((current_size - config['kernel_size'] + 2 * config['padding']) // config['stride']) + 1
            current_size = (current_size - config['pooling_size']) // config['pooling_size'] + 1
        fc_input_size = current_size * config['out_channels']
        print(fc_input_size)

        layers = []
        in_channels = config['n_sensors']
        for _ in range(config['n_layers']):
            layers += [
                nn.Conv1d(in_channels, config['out_channels'], config['kernel_size'], stride=config['stride'], padding=config['padding']),
                nn.BatchNorm1d(config['out_channels']),
                nn.LeakyReLU(negative_slope=config['negative_slope']),
                nn.MaxPool1d(config['pooling_size'])
            ]
            in_channels = config['out_channels']

        self.conv_layers = nn.Sequential(*layers)

        self.fc = nn.Linear(fc_input_size, config['num_classes'])

        self.train_loss = 0.0
        self.val_loss = 0.0
        self.val_progress = []

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_loss += loss.item()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)
        pred = F.softmax(logits, dim=1)
        self.val_loss += loss.item()
        self.val_progress.append((pred, y))
        return loss
    
    def on_validation_epoch_end(self):
        # calculate accuracy on full validation set
        preds = torch.cat([pred for pred, y in self.val_progress], dim=0)
        y = torch.cat([y for pred, y in self.val_progress], dim=0)
        preds = torch.argmax(preds, dim=1)
        acc = torch.sum(preds == y).item() / len(y)
        precision, recall, f1, _ = precision_recall_fscore_support(y.cpu(), preds.cpu(), average='macro')
        if self.config['log'] in ['wandb', 'all']:
            wandb.log({'train_loss': self.train_loss, 'val_loss': self.val_loss, 'val_acc': acc, 'val_precision': precision, 'val_recall': recall, 'val_f1': f1})
        if self.config['log'] in ['stdout', 'all']:
            print(f'Training loss: {self.train_loss}')
            print(f'Validation accuracy: {acc}')
            print(f'Validation loss: {self.val_loss}')
            print(f'Validation precision: {precision}')
            print(f'Validation recall: {recall}')
            print(f'Validation f1: {f1}')
        self.val_progress = []
        self.val_loss = 0.0
        self.train_loss = 0.0

    def configure_optimizers(self):
        # initialize Adam optimizer and scheduler with warmup
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config['lr'], steps_per_epoch=self.config['steps_per_epoch'], epochs=self.config['epochs'])
        return [self.optimizer], [self.scheduler]


def save_checkpoint(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'scheduler_state_dict': model.scheduler.state_dict()
    }, path)


def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])