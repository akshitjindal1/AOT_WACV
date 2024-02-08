from AOT_WACV.aot import *
import matplotlib.pyplot as plt
import plotly.express as px
import os
import plotly.io as pio
from sklearn.metrics import accuracy_score

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

def acc_and_agr(model_preds,gt,victim_preds):
    return accuracy_score(gt,model_preds), accuracy_score(victim_preds,model_preds)

def create_thief_loaders_ssl(thief_data, query_data, target_model, labeled_set, val_set, unlabeled_set, batch_size, idx_to_add_vs_labels):
    target_model.eval()
    ssl_indexes = list(idx_to_add_vs_labels.keys())
    with torch.no_grad():
        print("replacing labeled set labels with victim labels for train loader")
        temp_loader = DataLoader(Subset(query_data, labeled_set), batch_size=128,
                            pin_memory=False, num_workers=4, shuffle=False)
        mismatched_indexes = []
        cnt = 0
        for d, l0, ind0 in tqdm(temp_loader):
            d = d.cuda()
            labels = target_model(d).argmax(axis=1, keepdim=False)
            labels = labels.detach().cpu().numpy()
            ind0 = ind0.cpu().numpy()
            #replace victim labels with pseudo labels
            for i,idx in enumerate(ind0):
                if idx in ssl_indexes:
                    if idx_to_add_vs_labels[idx]!=labels[i]:
                        cnt+=1
                        mismatched_indexes.append([idx, idx_to_add_vs_labels[idx], labels[i]])
                        
#                         print(f"victim:{labels[i]}, committee:{idx_to_add_vs_labels[idx]}")
                    labels[i] = idx_to_add_vs_labels[idx]
            
            for ii, jj in enumerate(ind0):
                thief_data.samples[jj] = (thief_data.samples[jj][0], labels[ii])
        print(f"{cnt} pseudo labels don't match the victim labels out of {len(ssl_indexes)}")

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
    
    return train_loader, val_loader, unlabeled_loader, mismatched_indexes

if __name__ == "__main__":
    
    load_cfg_fom_args(description='Model Stealing')
    with open(os.path.join(cfg.SAVE_DIR,"cfg"),'rb') as f:
        cfg = pickle.load(f)
    
    testset, victim_normalization_transform, n_classes = load_victim_dataset(cfg.VICTIM.DATASET,cfg.VICTIM.ARCH)
    test_loader = DataLoader(testset, batch_size=128, num_workers=4, shuffle=False, pin_memory=False)  
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes\n")

    # Load victim model    
    target_model = load_victim_model(cfg.VICTIM.ARCH, cfg.VICTIM.PATH, n_classes)

    # Evaluate target model on target dataset: sanity check
    acc, f1 = testz(target_model, test_loader)
    print(f"Target model acc = {acc}\n")
    
    query_transform = transforms.Compose([transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201))])
    if cfg.THIEF.DATASET=='imagenet_full':
        query_transform = victim_normalization_transform
    query_data = load_thief_dataset(cfg.THIEF.DATASET,cfg.THIEF.DATA_ROOT,query_transform)

    # Setup validation, initial labeled, and unlabeled sets 
    torch.manual_seed(43)
    random.seed(43)
    n_cycles = cfg.ACTIVE.CYCLES            
    arch_list = cfg.THIEF.ARCH
    print(arch_list)
    with open(os.path.join(cfg.SAVE_DIR,f"labeled_set_trial_1_cycle_{n_cycles}"),'rb') as f:
        labeled_set = pickle.load(f)
    with open(os.path.join(cfg.SAVE_DIR,f"val_set_trial_1_cycle_{n_cycles}"),'rb') as f:
            val_set = pickle.load(f)
    with open(os.path.join(cfg.SAVE_DIR,f"unlabeled_set_trial_1_cycle_{n_cycles}"),'rb') as f:
            unlabeled_set = pickle.load(f)
    
    best_model_cycle = defaultdict()
    #load the best cycle for each ensemble member
    with open(os.path.join(cfg.SAVE_DIR, "best_model_cycle_dic.pkl"), 'rb') as f:
        best_model_cycle = pickle.load(f)
    
    model_pred_dict = defaultdict(list)
    for arch in arch_list:
        cycle = best_model_cycle[arch]
        pred_save_path  = os.path.join(cfg.SAVE_DIR, f'trial_1_cycle_{cycle}_{arch}_unlabeled_pred.dict')
        with open(pred_save_path,'rb') as f:
            model_pred_dict[arch] = pickle.load(f)
            
    idx_vs_pred = defaultdict(list)
    for arch, dic in model_pred_dict.items():
        print(f"\niterating over {arch}")
        for idx, proba in tqdm(dic.items()):
            if idx in unlabeled_set:
                idx_vs_pred[idx].append(proba)
    
    idx_to_add_vs_labels = defaultdict(int)
    for idx,pred in idx_vs_pred.items():
        labels = set()
        for probs in pred:
            labels.add(np.argmax(probs))
        if len(labels)==1:
            idx_to_add_vs_labels[idx] = labels.pop()
    
    label_counts = Counter(list(idx_to_add_vs_labels.values()))
    print(f'Label counts for pseudo labels')
    print(label_counts)
    
    labeled_set = np.append(labeled_set,list(idx_to_add_vs_labels.keys()))
    
    dic_ssl = {'MODEL':[],'ACC':[],'AGR':[]}
    ssl_models = defaultdict()
    for arch in arch_list:
        print(arch)
        model = CommitteeMember(arch,n_classes)
        thief_model = model.model
        testset_pretrained = load_victim_dataset(cfg.VICTIM.DATASET,arch,load_for_pretrained=True,pretrained_transform=model.transforms)
        test_loader_pretrained = DataLoader(testset_pretrained, batch_size=128, num_workers=4, shuffle=False, pin_memory=False)

        #load data for training the thief model
        thief_data = load_thief_dataset(cfg.THIEF.DATASET,cfg.THIEF.DATA_ROOT,model.transforms)

        cycle = best_model_cycle[arch]
        
        with open(os.path.join(cfg.SAVE_DIR,f"labeled_set_trial_1_cycle_{cycle}"),'rb') as f:
            labeled_set = pickle.load(f)
        with open(os.path.join(cfg.SAVE_DIR,f"val_set_trial_1_cycle_{cycle}"),'rb') as f:
                val_set = pickle.load(f)
        with open(os.path.join(cfg.SAVE_DIR,f"unlabeled_set_trial_1_cycle_{cycle}"),'rb') as f:
                unlabeled_set = pickle.load(f)
        
        print(f"Num labeled samples: {len(labeled_set)}")
        print(f"Num unlabeled samples: {len(unlabeled_set)}\n")
        
        ssl_indexes = list(idx_to_add_vs_labels.keys())
        labeled_set.extend(ssl_indexes)
        labeled_set = list(set(labeled_set))
        for i in ssl_indexes:
            unlabeled_set.remove(i)
        print(f"after SSL addition, U_L:{len(labeled_set)}, U_UL:{len(unlabeled_set)}\n")
        
        train_loader, val_loader, unlabeled_loader, mismatched_indexes = create_thief_loaders_ssl(thief_data, query_data, target_model, labeled_set, val_set, unlabeled_set, 128,idx_to_add_vs_labels)
        # train_loader_noaug, val_loader_noaug, unlabeled_loader_noaug,_ = create_thief_loaders_ssl(query_data, query_data,target_model, 
        #                                                                                         labeled_set, val_set, unlabeled_set,
        #                                                                                         128,idx_to_add_vs_labels)
        
        dataloaders  = {'train': train_loader, 'test': test_loader_pretrained, 'val': val_loader, 'unlabeled': unlabeled_loader}
        # dataloaders_noaug = {'train': train_loader_noaug, 'val': val_loader_noaug, 'unlabeled': unlabeled_loader_noaug}
        print("Dataloaders created")
        
        print("Training set distribution")
        print(dist(labeled_set, dataloaders['train']))
        print("\nValidation set distribution")
        print(dist(val_set,dataloaders['val']))
        
        print(f"Best cycle for model {arch} is {cycle}, loading saved model")
        save_path = os.path.join(cfg.SAVE_DIR, f'trial_1_cycle_{cycle}_{arch}_best.pth')
        state_dict = torch.load(save_path)['state_dict']
        # Load thief model
        thief_model.load_state_dict(state_dict) #SOMETIMES THE BETTER MODEL IS OBTAINED WHEN TRAINING FROM SCRATCH
        acc_test, _ = testz(thief_model, test_loader_pretrained)
        print(f"Before SSL test accuracy: {acc_test}\n")
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.SGD(thief_model.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
        save_path  = os.path.join(cfg.SAVE_DIR,f'{arch}_{best_model_cycle[arch]}_ssl.pth')
        
        print("Training Model with SSL")
        train_with_validation(thief_model, criterion, optimizer, scheduler, dataloaders, 100, 0, 0,save_path)
        acc, f1 = testz(thief_model, test_loader_pretrained)
        agr = agreement_diffaug(target_model, thief_model, test_loader,test_loader_pretrained)
        print(f'SSL model on target dataset: acc = {acc:.4f}, f1 = {f1:.4f}')
        ssl_models[arch] = thief_model
        dic_ssl['MODEL'].append(arch)
        dic_ssl['ACC'].append(acc)
        dic_ssl['AGR'].append(agr)
        
        #to avoid gpu fillup
        gc.collect()
        torch.cuda.empty_cache()
    
    print(dic_ssl)
    
    #find whether the model before or after SSL is better
    final_preds_all = defaultdict(list)
    for img,label in tqdm(test_loader):
        img = img.cuda()
        pred = target_model(img).argmax(axis=1,keepdim=False)
        pred = pred.detach().cpu().tolist()
        label=label.detach().cpu().tolist()
        final_preds_all['gt'].extend(label)
        final_preds_all['victim'].extend(pred)
        
    for arch in arch_list:
        save_path_ssl  = os.path.join(cfg.SAVE_DIR,f'{arch}_{best_model_cycle[arch]}_ssl.pth')
        model = CommitteeMember(arch,n_classes)
        thief_model = model.model
        testset_pretrained = load_victim_dataset(cfg.VICTIM.DATASET,arch,load_for_pretrained=True,
                                                pretrained_transform=model.transforms)
        test_loader_pretrained = DataLoader(testset_pretrained, batch_size=cfg.TRAIN.BATCH, num_workers=4,
                                            shuffle=False, pin_memory=False)
        state_dict = torch.load(save_path_ssl)['state_dict']
        thief_model.load_state_dict(state_dict)
        # print(testz(thief_model,test_loader_pretrained))
        for img,label in tqdm(test_loader_pretrained):
            img = img.cuda()
            pred = thief_model(img).argmax(axis=1,keepdim=False)
            pred = pred.detach().cpu().tolist()
            final_preds_all[arch].extend(pred)
    
    for arch in arch_list:
        print(arch)
        acc = accuracy_score(final_preds_all['gt'],final_preds_all[arch])
        agr = accuracy_score(final_preds_all['victim'],final_preds_all[arch])
        print(acc,agr)
        print()
    
    combined_preds = []
    for i in tqdm(range(len(testset))):
        all_model_preds_i = []
        for arch in arch_list:
            all_model_preds_i.append(final_preds_all[arch][i])
        counts = Counter(all_model_preds_i)
    #     import pdb;pdb.set_trace()
        final_pred = counts.most_common(1)[0][0]
        if counts[final_pred]<2:
            final_pred=-1
        combined_preds.append(final_pred)
    acc = accuracy_score(final_preds_all['gt'],combined_preds)
    agr = accuracy_score(final_preds_all['victim'],combined_preds)
    
    print("Ensemble accuracy")
    print(acc,agr)
            
    import pdb;pdb.set_trace()