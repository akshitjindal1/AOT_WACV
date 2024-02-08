import os
import gc
import sys
import random
import math
import csv
import numpy as np
from tqdm import tqdm
from tqdm import trange
import torch
import pandas as pd
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

from models.cifar10_models import resnet34 as cifar10_resnet34
from models.cifar import *
from train_utils import testz, train_with_validation, agree, dist, agreement_diffaug
from conf import cfg, load_cfg_fom_args
import datasets
from collections import Counter, defaultdict
from AOT_WACV.aot_models import CommitteeMember
from scipy.special import softmax
from scipy.stats import entropy
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_victim_dataset(dataset_name, model_arch,load_for_pretrained=False,pretrained_transform=None):
    test_datasets = datasets.__dict__.keys()
    if dataset_name not in test_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(test_datasets))
    dataset = datasets.__dict__[dataset_name]

    # load model family
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    print('modelfamily: ', modelfamily)

    if load_for_pretrained:
        if model_arch=='resnet32':
            test_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
        else:
            test_transform = pretrained_transform
        testset = dataset(cfg, train=False, transform=test_transform)
        return testset

    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']    
    testset = dataset(cfg, train=False, transform=test_transform) 
    n_classes = len(testset.classes)  

    return testset, test_transform, n_classes


def load_victim_model(arch, model_path, n_classes):
    # Define architecture
    if arch == 'cnn32':
        target_model = Simodel() 
    elif arch == 'resnet32':
        target_model = cifar10_resnet34(num_classes=n_classes)
    elif arch == 'resnet34':
        target_model = torch.hub.load("pytorch/vision:v0.14.1","resnet34")
        target_model.fc = torch.nn.Linear(512,n_classes)
    # Load weights
    try:
        state_dict = torch.load(model_path)['state_dict']
        state_dict = {key.replace("last_linear", "fc"): value for key, value in state_dict.items()}
        target_model.load_state_dict(state_dict, strict=False)
    except: 
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        target_model.load_state_dict(state_dict, strict=False)

    target_model=target_model.cuda()
    print(f"Loaded target model {model_path}")
    
    # Set victim model to use normalization transform internally
    # target_model = Victim(target_model, normalization_transform)
    
    return target_model
    
    
def load_thief_dataset(dataset_name, data_root,pretrained_transforms):
    if dataset_name == 'imagenet32':
        dataset = datasets.__dict__["ImageNet32"]                
        thief_data = dataset(data_root, transform=pretrained_transforms)
        
    elif dataset_name == 'imagenet_full':
        dataset = datasets.__dict__["ImageNet1k"]
        thief_data = dataset(cfg, transform=pretrained_transforms)
        
    else:
        raise AssertionError('invalid thief dataset')
    
    return thief_data
        
    
def create_thief_loaders(thief_data, query_data, target_model, labeled_set, val_set, unlabeled_set, batch_size):
    target_model.eval()
    with torch.no_grad():
        print("replacing labeled set labels with victim labels for train loader")
        temp_loader = DataLoader(Subset(query_data, labeled_set), batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=False)
        for d, l0, ind0 in tqdm(temp_loader):
            d = d.cuda()
            l = target_model(d).argmax(axis=1, keepdim=False)
            l = l.detach().cpu().tolist()
            for ii, jj in enumerate(ind0):
                thief_data.samples[jj] = (thief_data.samples[jj][0], l[ii])

        print("replacing val set labels with victim labels for val loader")
        temp_loader = DataLoader(Subset(query_data, val_set), batch_size=batch_size, 
                            pin_memory=False, num_workers=4, shuffle=False)
        for d,l,ind0 in tqdm(temp_loader):
            d = d.cuda()
            l = target_model(d).argmax(axis=1, keepdim=False)
            l = l.detach().cpu().tolist()
            # print(l)
            for ii, jj in enumerate(ind0):
                thief_data.samples[jj] = (thief_data.samples[jj][0], l[ii])

    train_loader = DataLoader(Subset(thief_data, labeled_set), batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=False)
    val_loader = DataLoader(Subset(thief_data, val_set), batch_size=batch_size, 
                            pin_memory=False, num_workers=4, shuffle=False)
    unlabeled_loader = DataLoader(Subset(thief_data, unlabeled_set), batch_size=batch_size, 
                                        pin_memory=False, num_workers=4, shuffle=False)
    
    return train_loader, val_loader, unlabeled_loader


if __name__ == "__main__":
    # import pdb;pdb.set_trace()
    load_cfg_fom_args(description='Model Stealing')
    with open(os.path.join(cfg.SAVE_DIR,"cfg"),'wb') as f: pickle.dump(cfg,f)
    # if cfg.THIEF.DATASET != 'imagenet32':
    #     torch.multiprocessing.set_start_method("spawn")
    
    # Load victim dataset (test split only)
    testset, victim_normalization_transform, n_classes = load_victim_dataset(cfg.VICTIM.DATASET, cfg.VICTIM.ARCH)
    test_loader = DataLoader(testset, batch_size=128, num_workers=4, shuffle=False, pin_memory=False)  
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes")

    # Load victim model    
    target_model = load_victim_model(cfg.VICTIM.ARCH, cfg.VICTIM.PATH, n_classes)

    # Evaluate target model on target dataset: sanity check
    acc, f1 = testz(target_model, test_loader)
    print(f"Target model acc = {acc}")
    
    # Begin trials
    li = []
    results_arr = []
    uncertainty_arr = []
    print("TRAINING COMMITTEE MEMBERS")

    for trial in range(cfg.TRIALS):
        #this is for cifar10
        query_transform = transforms.Compose([transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201))])
        if cfg.THIEF.DATASET=='imagenet_full':
            query_transform = victim_normalization_transform
        query_data = load_thief_dataset(cfg.THIEF.DATASET, cfg.THIEF.DATA_ROOT,query_transform)

        # Setup validation, initial labeled, and unlabeled sets 
        indices = list(range(min(cfg.THIEF.NUM_TRAIN, len(query_data))))
        seeds = [43, 42, 666, 111, 51]
        print(seeds[trial])
        torch.manual_seed(seeds[trial])
        random.seed(seeds[trial])
        random.shuffle(indices)
        # do not use the entire unlabeled set, use only SUBSET number of samples
        indices = indices[:cfg.THIEF.SUBSET]
        val_set = indices[:cfg.ACTIVE.VAL]
        labeled_set = indices[cfg.ACTIVE.VAL:cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL]
        unlabeled_set = indices[cfg.ACTIVE.VAL+cfg.ACTIVE.INITIAL:]
        
        
        
        for cycle in range(cfg.ACTIVE.CYCLES):
            for arch in cfg.THIEF.ARCH:
                save_path  = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_{arch}_best.pth')
                unlabeled_pred_save_path  = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_{arch}_unlabeled_pred.dict')
                if os.path.exists(save_path) and os.path.exists(unlabeled_pred_save_path):
                    continue
                with open(os.path.join(cfg.SAVE_DIR,f"labeled_set_trial_{trial+1}_cycle_{cycle+1}"),'wb') as f: pickle.dump(labeled_set,f)
                with open(os.path.join(cfg.SAVE_DIR,f"unlabeled_set_trial_{trial+1}_cycle_{cycle+1}"),'wb') as f: pickle.dump(unlabeled_set,f)
                with open(os.path.join(cfg.SAVE_DIR,f"val_set_trial_{trial+1}_cycle_{cycle+1}"),'wb') as f: pickle.dump(val_set,f)
                print(f"Starting training of {arch}")
                model = CommitteeMember(arch,n_classes)
                thief_model = model.model
                # import pdb;pdb.set_trace()
                #load testset with appropriate transformations to work with thief model
                testset_pretrained = load_victim_dataset(cfg.VICTIM.DATASET,arch,load_for_pretrained=True,pretrained_transform=model.transforms)
                test_loader_pretrained = DataLoader(testset_pretrained, batch_size=128, num_workers=4, shuffle=False, pin_memory=False)
                
                #load data for training the thief model
                thief_data = load_thief_dataset(cfg.THIEF.DATASET, cfg.THIEF.DATA_ROOT,model.transforms)
                #data used to query the target model has to be without augmentations
                print(f"Num labeled samples: {len(labeled_set)}")
                print(f"Num unlabeled samples: {len(unlabeled_set)}")

                # Create train, val and unlabeled dataloaders
                train_loader, val_loader, unlabeled_loader = create_thief_loaders(thief_data, query_data, target_model, labeled_set, val_set, unlabeled_set, cfg.TRAIN.BATCH)
                train_loader_noaug, val_loader_noaug, unlabeled_loader_noaug = create_thief_loaders(query_data, query_data,target_model, labeled_set, val_set, unlabeled_set, cfg.TRAIN.BATCH)
                dataloaders  = {'train': train_loader, 'test': test_loader_pretrained, 'val': val_loader, 'unlabeled': unlabeled_loader}
                dataloaders_noaug = {'train': train_loader_noaug, 'val': val_loader_noaug, 'unlabeled': unlabeled_loader_noaug}
                print("Dataloaders created")
                # import pdb;pdb.set_trace()
                print('Validation set distribution: ')
                val_dist =   dist(val_set, dataloaders['val'])
                print(val_dist)

                print('Labeled set distribution: ')
                label_dist = dist(labeled_set, dataloaders['train'])
                print(label_dist)
                # Load thief model
                # Compute metrics on test dataset
                acc, f1 = testz(thief_model, test_loader_pretrained)
                agr = agreement_diffaug(target_model, thief_model, test_loader,test_loader_pretrained)
                print(f'Initial model on test set: acc = {acc:.4f}, agreement = {agr:.4f}, f1 = {f1:.4f}')
                
                # Compute metrics on validation dataset
                acc, f1 = testz(thief_model, dataloaders['val'])
                agr = agreement_diffaug(target_model, thief_model,dataloaders_noaug['val'], dataloaders['val'])
                print(f'Initial model on validation dataset: acc = {acc:.4f}, agreement = {agr:.4f}, f1 = {f1:.4f}')

                # Set up thief optimizer, scheduler
                criterion = nn.CrossEntropyLoss(reduction='none')
                learning_rate = 0.01 if arch=='alexnet' else cfg.TRAIN.LR
                if cfg.TRAIN.OPTIMIZER == 'SGD':
                    optimizer = optim.SGD(thief_model.parameters(), lr=learning_rate, 
                                            momentum=cfg.TRAIN.MOMENTUM,weight_decay=cfg.TRAIN.WDECAY)
                elif cfg.TRAIN.OPTIMIZER == 'Adam':
                    optimizer = optim.Adam(thief_model.parameters(), lr=learning_rate, weight_decay=cfg.TRAIN.WDECAY)
                else:
                    raise AssertionError('Unknown optimizer')
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)
                # Train thief model for current cycle
                save_path  = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_{arch}_best.pth')
                if not os.path.exists(save_path):
                    print("Training new Model")
                    train_with_validation(thief_model, criterion, optimizer, scheduler, dataloaders, cfg.TRAIN.EPOCH, trial, cycle,save_path)
                else:
                    print("Load best checkpoint for thief model. Recalculating performance on test set")
                    best_state = torch.load(save_path)['state_dict']
                    thief_model.load_state_dict(best_state)
                    
                    # Compute accuracy and agreement on target dataset
                    acc, f1 = testz(thief_model, test_loader_pretrained)
                    agr = agreement_diffaug(target_model, thief_model, test_loader,test_loader_pretrained)
                    print(f'Initial model on target dataset: acc = {acc:.4f}, agreement = {agr:.4f}, f1 = {f1:.4f}')
                # import pdb;pdb.set_trace()
                labeled_pred_dict, unlabeled_pred_dict = dict(), dict()
                labeled_pred_save_path  = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_{arch}_labeled_pred.dict')
                unlabeled_pred_save_path  = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_{arch}_unlabeled_pred.dict')
                
                print("saving labeled set predictions")
                if not os.path.exists(labeled_pred_save_path):
                    for img,label,idx in tqdm(dataloaders['train']):
                        img = img.cuda()
                        pred = thief_model(img)
                        idx = idx.cpu().detach().numpy()
                        pred = pred.cpu().detach().numpy()
                        for i,index in enumerate(idx):
                            if cfg.ACTIVE.METHOD == 'qbc+kcenter':
                                labeled_pred_dict[index] = pred[i]    
                            else:
                                labeled_pred_dict[index] = softmax(pred[i]).round(4)
                    with open(labeled_pred_save_path,'wb') as f:
                        pickle.dump(labeled_pred_dict,f)
                else:
                    with open(labeled_pred_save_path,'rb') as f:
                        labeled_pred_dict = pickle.load(f)
                
                print("saving unlabeled set predictions")
                if not os.path.exists(unlabeled_pred_save_path):
                    for img,label,idx in tqdm(dataloaders['unlabeled']):
                        img = img.cuda()
                        pred = thief_model(img)
                        idx = idx.cpu().detach().numpy()
                        pred = pred.cpu().detach().numpy()
                        for i,index in enumerate(idx):
                            if cfg.ACTIVE.METHOD == 'qbc+kcenter':
                                unlabeled_pred_dict[index] = pred[i]    
                            else:
                                unlabeled_pred_dict[index] = softmax(pred[i]).round(4)
                    with open(unlabeled_pred_save_path,'wb') as f:
                        pickle.dump(unlabeled_pred_dict,f)
                else:
                    with open(unlabeled_pred_save_path,'rb') as f:
                        unlabeled_pred_dict = pickle.load(f)
                
                print(f"Saved {arch} predictions to file")
                acc, f1 = testz(thief_model, test_loader_pretrained)
                agr = agreement_diffaug(target_model, thief_model, test_loader,test_loader_pretrained)
                
                #freeing memory
                thief_model.cpu()
                del model, thief_model, unlabeled_pred_dict, labeled_pred_dict
                gc.collect()
                torch.cuda.empty_cache()

                print('Trial {}/{} || Cycle {}/{} || Label set size {} || Test acc {} || Test agreement {}'.format(trial+1, cfg.TRIALS, cycle+1, cfg.ACTIVE.CYCLES, len(labeled_set), acc, agr))
                print("*"*100, "\n")

            # TODO: generate and store predictions of every committee member on the test dataset, uncertainity and final label etc.
            # after every model has been trained and predictions stored along with confidence, use a policy to determine the most uncertain samples to query            
            if cycle!=cfg.ACTIVE.CYCLES-1:
                if cfg.ACTIVE.METHOD == 'random':
                    continue
                if cfg.ACTIVE.METHOD == 'qbc':
                    #load all saved prediction files
                    model_pred_dict = dict()
                    for arch in cfg.THIEF.ARCH:
                        pred_save_path  = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_{arch}_unlabeled_pred.dict')
                        with open(pred_save_path,'rb') as f:
                            model_pred_dict[arch] = pickle.load(f)
                    idx_vs_pred = defaultdict(list)
                    for arch, dic in model_pred_dict.items():
                        print(f"iterating over {arch}")
                        for idx, proba in tqdm(dic.items()):
                            idx_vs_pred[idx].append(proba)
                    uncertainty,indexes = [],[]
                    for idx,pred_list in idx_vs_pred.items():
                        indexes.append(idx)
                        uncertainty.append(entropy(np.array(pred_list).sum(axis=0)))
                    arg = np.argsort(np.array(uncertainty))
                    selected_index_list = np.array(indexes)[arg][-(cfg.ACTIVE.ADDENDUM):]
                
                elif cfg.ACTIVE.METHOD == 'qbc_vote':
                    #load all saved prediction files
                    model_pred_dict = dict()
                    for arch in cfg.THIEF.ARCH:
                        pred_save_path = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_{arch}_unlabeled_pred.dict')
                        with open(pred_save_path,'rb') as f:
                            model_pred_dict[arch] = pickle.load(f)
                    idx_vs_pred = defaultdict(list)
                    for arch, dic in model_pred_dict.items():
                        print(f"iterating over {arch}")
                        for idx, proba in tqdm(dic.items()):
                            idx_vs_pred[idx].append(np.argmax(proba))

                    uncertainty,indexes = [],[]
                    for idx,pred_list in idx_vs_pred.items():
                        indexes.append(idx)
                        class_counts = Counter(pred_list)
                        class_dist = [class_counts[c]/len(pred_list) for c in range(10)]
                        uncertainty.append(entropy(np.array(class_dist)))
                    arg = np.argsort(np.array(uncertainty))
                    selected_index_list = np.array(indexes)[arg][-(cfg.ACTIVE.ADDENDUM):]
                
                elif cfg.ACTIVE.METHOD == 'qbc+kcenter':
                    #load all saved prediction files
                    model_pred_dict_labeled, model_pred_dict_unlabeled = dict(), dict()
                    for arch in cfg.THIEF.ARCH:
                        labeled_pred_save_path  = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_{arch}_labeled_pred.dict')
                        unlabeled_pred_save_path  = os.path.join(cfg.SAVE_DIR, f'trial_{trial+1}_cycle_{cycle+1}_{arch}_unlabeled_pred.dict')
                        with open(labeled_pred_save_path,'rb') as f:
                            model_pred_dict_labeled[arch] = pickle.load(f)
                        with open(unlabeled_pred_save_path,'rb') as f:
                            model_pred_dict_unlabeled[arch] = pickle.load(f)
                    
                    idx_vs_pred_unlabeled = defaultdict(list)
                    for arch, dic in model_pred_dict_unlabeled.items():
                        print(f"iterating over {arch}")
                        for idx, proba in tqdm(dic.items()):
                            idx_vs_pred_unlabeled[idx].append(proba)

                    #apply kcenter first
                    # selected_idx_vs_dist = defaultdict(list)

                    # for arch in cfg.THIEF.ARCH:
                    #     import pdb;pdb.set_trace()
                    #     print(f"iterating over {arch}")
                    #     labeled_set_preds = np.array(list(model_pred_dict_labeled[arch].values()))
                    #     unlabeled_set_preds = np.array(model_pred_dict_unlabeled[arch].values())
                    #     dists = []
                    #     num_batches = math.ceil(len(unlabeled_set_preds)/10000)
                    #     for batch in range(num_batches):
                    #         dists.extend(torch.cdist(torch.tensor(unlabeled_set_preds[batch*10000:(batch+1)*10000]),torch.tensor(labeled_set_preds)).min(axis=1).values.cpu().numpy())
                    #     for i,idx in enumerate(model_pred_dict_unlabeled[arch].keys()):
                    #         selected_idx_vs_dist[idx].append(dists[i])
                    
                    # for idx, dist_vector in selected_idx_vs_dist.items():
                    #     selected_idx_vs_dist[idx] = np.mean(dist_vector)
                    
                    # selected_index_list = list(dict(sorted(selected_idx_vs_dist.items(), key=lambda item: -1*item[1])).keys())[:10000]

                    # #apply consensus on kcenter output
                    # uncertainty,indexes = [],[]
                    # for idx in selected_index_list:
                    #     indexes.append(idx)
                    #     uncertainty.append(entropy(np.array(model_pred_dict_unlabeled[idx]).sum(axis=0)))
                    # arg = np.argsort(np.array(uncertainty))
                    # selected_index_list = np.array(indexes)[arg][-cfg.ACTIVE.ADDENDUM:]


                    # #apply qbc consensus function
                    # uncertainty,indexes = [],[]
                    # for idx,pred_list in idx_vs_pred_unlabeled.items():
                    #     indexes.append(idx)
                    #     uncertainty.append(entropy(np.array(pred_list).sum(axis=0)))
                    # arg = np.argsort(np.array(uncertainty))
                    # selected_index_list = np.array(indexes)[arg][-200000:]

                    # #apply kcenter on the selected_index_list
                    # selected_idx_vs_dist = defaultdict(list)

                    # for arch in cfg.THIEF.ARCH:
                    #     print(f"iterating over {arch}") 
                    #     labeled_set_preds = np.array(list(model_pred_dict_labeled[arch].values()))
                    #     unlabeled_set_preds = np.array([model_pred_dict_unlabeled[arch][i] for i in selected_index_list])
                    #     dists = torch.cdist(torch.tensor(unlabeled_set_preds),torch.tensor(labeled_set_preds)).min(axis=1).values.cpu().numpy()
                    #     for i,idx in enumerate(selected_index_list):
                    #         selected_idx_vs_dist[idx].append(dists[i])  
                    
                    # for idx, dist_vector in selected_idx_vs_dist.items():
                    #     selected_idx_vs_dist[idx] = np.mean(dist_vector)
                    
                    # selected_index_list = list(dict(sorted(selected_idx_vs_dist.items(), key=lambda item: -1*item[1])).keys())[:cfg.ACTIVE.ADDENDUM]
                    
                    #KMEANS++++ 
                    #apply qbc consensus function
                    uncertainty,indexes = [],[]
                    for idx,pred_list in idx_vs_pred_unlabeled.items():
                        indexes.append(idx)
                        uncertainty.append(entropy(np.array(pred_list).sum(axis=0)))
                    arg = np.argsort(np.array(uncertainty))
                    qbc_index_list = np.array(indexes)[arg][-200000:]

                    #apply kcenter on the selected_index_list
                    selected_index_list = []
                    idx_vs_dists = defaultdict(lambda: [np.inf]*len(cfg.THIEF.ARCH))
                    selected_idx_vs_dist = defaultdict(float)
                    
                    # import pdb;pdb.set_trace()
                    print("starting kcenter")
                    for _ in trange(cfg.ACTIVE.ADDENDUM):
                        for ind,arch in enumerate(cfg.THIEF.ARCH):
                            # print(f"iterating over {arch}")
                            if len(selected_index_list)!=0:
                                labeled_set_preds = np.array([model_pred_dict_unlabeled[arch][i] for i in selected_index_list])
                            else:
                                labeled_set_preds = np.array(list(model_pred_dict_labeled[arch].values()))
                            # print(labeled_set_preds.shape)
                            unlabeled_set_preds = np.array([model_pred_dict_unlabeled[arch][i] for i in qbc_index_list])
                            
                            dists = torch.cdist(torch.tensor(unlabeled_set_preds),torch.tensor(labeled_set_preds)).min(axis=1).values.cpu().numpy()
                            for i,idx in enumerate(qbc_index_list):
                                idx_vs_dists[idx][ind] = min(dists[i],idx_vs_dists[idx][ind])
                            # pdb.set_trace()
                        dists = [np.mean(idx_vs_dists[idx]) for idx in qbc_index_list]
                        selected_index = qbc_index_list[np.argmax(dists)]
                        
                        selected_index_list.append(selected_index)
                        qbc_index_list = np.delete(qbc_index_list,np.argmax(dists))
                    
                    
                    
                else:
                    raise(AssertionError)
                
                l = list(selected_index_list)
                labeled_set += l
                unlabeled_set = [unlabeled_set[x] for x in range(len(unlabeled_set)) if unlabeled_set[x] not in selected_index_list]

        
        

    import pdb;pdb.set_trace()