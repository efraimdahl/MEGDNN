from utils import read_datasets, MEGDataset, fit_transform_scaler, temporal_downsampling, batchify_activity, error_analysis
from models import MEGConvNet, MEGNet, LSTMNet, load_checkpoint, save_checkpoint
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
import wandb
import yaml
from time import time
import os


if __name__ == '__main__':
    # read config file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        print(f'successfully loaded config file {args.config}')
    
    # read datasets
    # creating a dictionary to encode the different tasks
    data = read_datasets(config['data_dir'])
    encoder_dict = {
        'rest': 0,
        'task_motor': 1,
        'task_story_math': 2,
        'task_working_memory': 3,
    }
    decoder_dict = {v: k for k, v in encoder_dict.items()}

    # login to wandb via API key
    if config['log'] in ['wandb', 'all']:
        wandb.login(key = config['wandb_api_key'])
    
    # check if folder 'weights' exists and create if not
    if not os.path.exists('weights'):
        os.makedirs('weights')


    # preprocess data and train model on intra-subject data
    if 'intra' in config['validation']:
        # initialize wandb
        if config['log'] in ['wandb', 'all']:
            print('Input wandb run name for intra-subject run:')
            run_name = input()
            run_name = 'intra_'+run_name
            config['run_name'] = run_name
            wandb.init(project='MEG', config=config, name = run_name)

        # Extract test and training data from the dataset
        X_train, y_train = data['intra']['X_train'], data['intra']['y_train']
        X_test, y_test = data['intra']['X_test'], data['intra']['y_test']
        y_train = [encoder_dict[label] for label in y_train]
        y_test = [encoder_dict[label] for label in y_test]

        # Scale the data
        X_train_scaled, X_test_scaled = fit_transform_scaler(X_train, [X_test], scaler=StandardScaler())
        X_test_scaled = X_test_scaled[0]

        # Downsample using the configurated downsampling method
        X_train = temporal_downsampling(X_train_scaled, downsample_factor=config['downsample'])
        X_test = temporal_downsampling(X_test_scaled, downsample_factor=config['downsample'])
        
        # Split samples into batches of size window_size
        # To run the experiment without batchifying, initiate window_size at -1
        if config['window_size'] != -1:
            X_train, y_train = batchify_activity(X_train, y_train, window_size = config['window_size'])
            X_test, y_test = batchify_activity(X_test, y_test, window_size = config['window_size'])
        
        if config['model'] != 'MEGConvNet' and config['model']!="LSTMNet":
            X_train = [x.flatten() for x in X_train]
            X_test = [x.flatten() for x in X_test]
        
        # defines the training and validation sets
        train_set = MEGDataset(X_train, y_train)
        val_set = MEGDataset(X_test, y_test)

        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'])

        # initialize the model based on the config file
        if config['model'] == 'MEGConvNet':
            model = MEGConvNet(config)
        elif config['model'] == 'MEGNet':
            model = MEGNet(config)
        elif config['model'] == "LSTMNet":
            model = LSTMNet(config)
        
        #Gives the output of the models and logs them to wandb
        print(f"model: {config['model']}")
        print('Training on intra-subject data...')
        trainer = pl.Trainer(max_epochs=config['epochs'], log_every_n_steps=2)

        start = time()
        trainer.fit(model, train_loader, val_loader)
        end = time()
        wandb.log({'INTRA training_time': end - start, 'INTRA average train+val time per epoch': (end - start) / config['epochs']})
        if config['log_best']:
            if config['target_metric'] == 'accuracy':
                wandb.log({'INTRA best accuracy': model.best_acc})
            elif config['target_metric'] == 'f1':
                wandb.log({'INTRA best f1': model.best_f1})
        if not config['save_best']:
            save_checkpoint(model, 'weights/'+run_name+'.pth')
        wandb.finish()
    
    # preprocess data and train model on cross-subject data
    if 'cross' in config['validation']:
        if config['log'] in ['wandb', 'all']:
            # initialize wandb
            print('Input wandb run name for cross-subject run:')
            if(config["name"]):
                run_name=config["name"]
            else:
                run_name = input()
            run_name = 'cross_'+run_name
            config['run_name'] = run_name
            wandb.init(project='MEG', config=config, name = run_name)
        
        # Extract test and training data from the dataset
        X_train, y_train = data['cross']['X_train'], data['cross']['y_train']
        X_test = [data['cross']['X_test1'], data['cross']['X_test2'], data['cross']['X_test3']]
        y_test = [data['cross']['y_test1'], data['cross']['y_test2'], data['cross']['y_test3']]

        y_train = [encoder_dict[label] for label in y_train]
        y_test = [[encoder_dict[label] for label in y] for y in y_test]

        # Scale the data
        X_train_scaled, X_test_scaled = fit_transform_scaler(X_train, X_test, scaler=StandardScaler())

        # Downsample using the configurated downsampling method
        X_train = temporal_downsampling(X_train_scaled, downsample_factor=config['downsample'])
        X_test = [temporal_downsampling(x, downsample_factor=config['downsample']) for x in X_test_scaled]
        
        # merges the different test sets into one for validation callback
        X_test_merged = []
        y_test_merged = []
        for i in range(len(X_test)):
            X_test_merged.extend(X_test[i])
            y_test_merged.extend(y_test[i])
        
        # Split samples into batches of size window_size
        # To run the experiment without batchifying, initiate window_size at -1
        if config['window_size'] != -1:
            X_train, y_train = batchify_activity(X_train, y_train, window_size = config['window_size'])
            X_test_merged, y_test_merged = batchify_activity(X_test_merged, y_test_merged, window_size = config['window_size'])
        
        # Flattens the data
        if config['model'] != 'MEGConvNet' and config['model']!="LSTMNet":
            X_train = [x.flatten() for x in X_train]
            X_test_merged = [x.flatten() for x in X_test_merged]
        
        # defines the training and validation sets
        train_set = MEGDataset(X_train, y_train)
        val_set = MEGDataset(X_test_merged, y_test_merged)

        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'])

        # initialize the model based on the config file
        if config['model'] == 'MEGConvNet':
            model = MEGConvNet(config)
        elif config['model'] == 'MEGNet':
            model = MEGNet(config)
        elif config['model'] == 'LSTMNet':
            model = LSTMNet(config)
        
        #Gives the output of the models and logs them to wandb
        print(f"model: {config['model']}")
        print('Training on cross-subject data...')
        trainer = pl.Trainer(max_epochs=config['epochs'], log_every_n_steps=2)

        start = time()
        trainer.fit(model, train_loader, val_loader)
        end = time()
        # logs average training time per epoch and total training time
        wandb.log({'CROSS training_time': end - start, 'CROSS average train+val time per epoch': (end - start) / config['epochs']})

        # logs best accuracy or f1 score
        if config['log_best']:
            if config['target_metric'] == 'accuracy':
                wandb.log({'CROSS best accuracy': model.best_acc})
            elif config['target_metric'] == 'f1':
                wandb.log({'CROSS best f1': model.best_f1})
        if not config['save_best']:
            save_checkpoint(model, 'weights/'+run_name+'.pth')
        
        # performs error analysis on the cross test sets
        if config['error_analysis']:
            # if save_best is true, load the best model for validation
            if config['save_best']:
                load_checkpoint(model, 'weights/'+run_name+'.pth')
            error_analysis(model, data, encoder_dict, config)
        wandb.finish()
