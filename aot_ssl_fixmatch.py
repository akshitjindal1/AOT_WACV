from AOT_WACV.aot import *
import matplotlib.pyplot as plt
import plotly.express as px
import os
import plotly.io as pio
from sklearn.metrics import accuracy_score
from Fixmatch.randaugment import RandAugmentMC

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

def create_thief_loaders_ssl_fixmatch(thief_data, query_data, thief_data_strong_aug, target_model, labeled_set, val_set, 
                                      unlabeled_set, batch_size, idx_to_add_vs_labels):
    target_model.eval()
    ssl_indexes = list(idx_to_add_vs_labels.keys())
    with torch.no_grad():
        print("replacing labeled set labels with victim labels for train loader")
        temp_loader = DataLoader(Subset(query_data, labeled_set), batch_size=128,
                            pin_memory=False, num_workers=4, shuffle=False)
        cnt = 0
        for d, l0, ind0 in tqdm(temp_loader):
            d = d.cuda()
            labels = target_model(d).argmax(axis=1, keepdim=False)
            labels = labels.detach().cpu().numpy()
            ind0 = ind0.cpu().numpy()
            for ii, jj in enumerate(ind0):
                thief_data.samples[jj] = (thief_data.samples[jj][0], labels[ii])
        
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
        
        print("replacing unlabeled set labels with pseudo labels for unlabeled loader")
        
        for idx in ssl_indexes:
            thief_data_strong_aug.samples[idx] = (thief_data_strong_aug.samples[idx][0],idx_to_add_vs_labels[idx])
    
    train_loader = DataLoader(Subset(thief_data, labeled_set), batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=False)
    val_loader = DataLoader(Subset(thief_data, val_set), batch_size=batch_size, 
                            pin_memory=False, num_workers=4, shuffle=False)
    unlabeled_loader = DataLoader(Subset(thief_data_strong_aug, unlabeled_set), batch_size=batch_size, 
                                        pin_memory=False, num_workers=4, shuffle=False)
    
    return train_loader, val_loader, unlabeled_loader

def train_with_validation_fixmatch(model, criterion, optimizer, scheduler, dataloaders, num_epochs, trial, cycle, out_dir,
                                   display_every = 10, early_stop_tolerance=10):
    print('>> Train a Model.')
    
    exit = False
    curr_loss = None
    best_f1 = None
    no_improvement = 0
    
    train_acc, train_f1 = testz(model, dataloaders['train'])
    val_acc, val_f1 = testz(model, dataloaders['val'])
    test_acc, test_f1 = testz(model, dataloaders['test'])

    print(f"Trial {trial+1}, Cycle {cycle+1}")
    print(f"Train acc/f1 = {train_acc:.4f} / {train_f1:.4f}")
    print(f"Val acc/f1 = {val_acc:.4f} / {val_f1:.4f}")
    print(f"Test acc/f1 = {test_acc:.4f} / {test_f1:.4f}")
    

    for epoch in tqdm(range(num_epochs), leave=False):
        model.train()
        # global iters
        iters = 0
        total_loss = 0
        for data in dataloaders['train']:
            inputs = data[0].cuda()
            target = data[1].cuda()
            iters += 1

            optimizer.zero_grad()
            output = model(inputs)
            target_loss = criterion(output, target)

            loss = torch.sum(target_loss) / target_loss.size(0)
            total_loss += torch.sum(target_loss)

            loss.backward()
            optimizer.step()

        for data in dataloaders['unlabeled']:
            inputs = data[0].cuda()
            target = data[1].cuda()
            iters += 1

            optimizer.zero_grad()
            output = model(inputs)
            target_loss = criterion(output, target)

            loss = torch.sum(target_loss) / target_loss.size(0)
            total_loss += torch.sum(target_loss)

            loss.backward()
            optimizer.step()
        
        mean_loss = total_loss / iters
        # import pdb;pdb.set_trace()
        scheduler.step()
        model.eval()
        if (epoch+1)%2==0:
            
            val_acc, val_f1 = testz(model, dataloaders['val'])

            if best_f1 is None or val_f1 > best_f1 :
                best_f1 = val_f1
                torch.save({
                    'trial': trial + 1,
                    'cycle': cycle + 1, 
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    },
                    out_dir)
                no_improvement = 0
            else:
                no_improvement += 1
                if (no_improvement % early_stop_tolerance) == 0:
                    exit = True

        # Display progress
        if (epoch+1) % display_every == 0:
            train_acc, train_f1 = testz(model, dataloaders['train'])
            test_acc, test_f1 = testz(model, dataloaders['test'])
            print(f"Epoch {epoch+1}: Train acc/f1 = {train_acc:.4f} / {train_f1:.4f} \n\
                Val acc/f1 = {val_acc:.4f} / {val_f1:.4f} \n\
                Test acc/f1 = {test_acc:.4f} / {test_f1:.4f}")

        if exit:
            print(f"Number of epochs processed: {epoch+1} in cycle {cycle+1}") 
            break

    train_acc, train_f1 = testz(model, dataloaders['train'])
    val_acc, val_f1 = testz(model, dataloaders['val'])
    test_acc, test_f1 = testz(model, dataloaders['test'])

    print(f"Trial {trial+1}, Cycle {cycle+1}")
    print(f"Train acc/f1 = {train_acc:.4f} / {train_f1:.4f}")
    print(f"Val acc/f1 = {val_acc:.4f} / {val_f1:.4f}")
    print(f"Test acc/f1 = {test_acc:.4f} / {test_f1:.4f}")

    print('>> Finished.')
    
    

if __name__ == "__main__":
    
    load_cfg_fom_args(description='Model Stealing')
    with open(os.path.join(cfg.SAVE_DIR,"cfg"),'rb') as f:
        cfg = pickle.load(f)
    
    testset, victim_normalization_transform, n_classes = load_victim_dataset(cfg.VICTIM.DATASET,cfg.VICTIM.ARCH)
    test_loader = DataLoader(testset, batch_size=128, num_workers=4, shuffle=False, pin_memory=False)  
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes")

    # Load victim model    
    target_model = load_victim_model(cfg.VICTIM.ARCH, cfg.VICTIM.PATH, n_classes)

    # Evaluate target model on target dataset: sanity check
    acc, f1 = testz(target_model, test_loader)
    print(f"Target model acc = {acc}")
    
    query_transform=None
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
    
    dic_ssl = {'MODEL':[],'ACC':[],'AGR':[]}
    ssl_models = defaultdict()
    best_model_cycle_preds = defaultdict(list)
    best_model_cycle_preds_wkaug = defaultdict(list)

    if os.path.exists(os.path.join(cfg.SAVE_DIR,'best_model_preds.dict')) and os.path.exists(os.path.join(cfg.SAVE_DIR,'best_model_preds_wkaug.dict')):
        print("LOADING PRE SAVED BEST MODEL PREDICTIONS")
        with open(os.path.join(cfg.SAVE_DIR,'best_model_preds.dict'), 'rb') as f:
            best_model_cycle_preds = pickle.load(f)
        with open(os.path.join(cfg.SAVE_DIR,'best_model_preds_wkaug.dict'), 'rb') as f:
            best_model_cycle_preds_wkaug = pickle.load(f)
    else:
        for arch in arch_list:
            model = CommitteeMember(arch,n_classes)
            thief_model = model.model
            best_cycle = best_model_cycle[arch]
            print(f"\nBest {arch} is of cycle {best_cycle}")
            save_path = os.path.join(cfg.SAVE_DIR,f'trial_1_cycle_{cycle}_{arch}_best.pth')
            state_dict = torch.load(save_path)['state_dict']
            thief_model.load_state_dict(state_dict)
            
            testset_pretrained = load_victim_dataset(cfg.VICTIM.DATASET,arch,load_for_pretrained=True,pretrained_transform=model.transforms)
            test_loader_pretrained = DataLoader(testset_pretrained, batch_size=128, num_workers=4, shuffle=False, pin_memory=False)
            acc_test,_ = testz(thief_model,test_loader_pretrained)
            print(f"Best model accuracy for {arch}: {acc_test}\n")
            
            weak_transform = transforms.RandomHorizontalFlip(p=1.0)
            model_transform_wk = transforms.Compose([weak_transform, model.transforms])
            
            testset_pretrained_wk = load_victim_dataset(cfg.VICTIM.DATASET, arch, load_for_pretrained=True,pretrained_transform = model_transform_wk)
            test_loader_pretrained_wk = DataLoader(testset_pretrained_wk, batch_size=cfg.TRAIN.BATCH, num_workers=4,shuffle=False, pin_memory=False)
            acc_test_wk,_ = testz(thief_model, test_loader_pretrained_wk)
            print(f"Thief model acc without weak aug: {acc_test}, with weak aug: {acc_test_wk}")

            with open(os.path.join(cfg.SAVE_DIR,"unlabeled_set_trial_1_cycle_10"),'rb') as f:
                unlabeled_set = pickle.load(f)
            
            print("getting normal labels")
            thief_data = load_thief_dataset(cfg.THIEF.DATASET, cfg.THIEF.DATA_ROOT, model.transforms)
            thief_loader = DataLoader(Subset(thief_data, unlabeled_set), batch_size=128, pin_memory=False,num_workers=4, shuffle=False)
            for img,label,idx in tqdm(thief_loader):
                img = img.cuda()
                probs = thief_model(img)
                labels = probs.argmax(axis=1, keepdim=False)
                labels = labels.detach().cpu().numpy()
                probs = probs.detach().cpu().numpy()
                probs = np.divide(np.exp(probs), np.sum(np.exp(probs), axis=1, keepdims=True))
                idx = idx.cpu().numpy()
                confs = probs[np.arange(labels.shape[0]),labels]
                best_model_cycle_preds[f'{arch}_preds'].extend(list(labels))
                best_model_cycle_preds[f'{arch}_confs'].extend(list(confs))
            
            print("getting weak augmentation labels")
            thief_data = load_thief_dataset(cfg.THIEF.DATASET, cfg.THIEF.DATA_ROOT, model_transform_wk)
            thief_loader = DataLoader(Subset(thief_data, unlabeled_set), batch_size=128, pin_memory=False,num_workers=4, shuffle=False)
            for img,label,idx in tqdm(thief_loader):
                img = img.cuda()
                probs = thief_model(img)
                labels = probs.argmax(axis=1, keepdim=False)
                labels = labels.detach().cpu().numpy()
                probs = probs.detach().cpu().numpy()
                probs = np.divide(np.exp(probs), np.sum(np.exp(probs), axis=1, keepdims=True))
                idx = idx.cpu().numpy()
                confs = probs[np.arange(labels.shape[0]),labels]
                best_model_cycle_preds_wkaug[f'{arch}_preds'].extend(list(labels))
                best_model_cycle_preds_wkaug[f'{arch}_confs'].extend(list(confs))
        
        with open(os.path.join(cfg.SAVE_DIR,'best_model_preds.dict'), 'wb') as f:
            pickle.dump(best_model_cycle_preds,f)
        with open(os.path.join(cfg.SAVE_DIR,'best_model_preds_wkaug.dict'), 'wb') as f:
            pickle.dump(best_model_cycle_preds_wkaug,f)   
        
    selected_samples_per_model = defaultdict(lambda: defaultdict(list))
    mismatch_set = defaultdict(list)
    for arch in cfg.THIEF.ARCH:
        match_count, mismatch_count = 0, 0
        for idx in tqdm(range(len(best_model_cycle_preds[f'{arch}_preds']))):
            if best_model_cycle_preds[f'{arch}_preds'][idx] == best_model_cycle_preds_wkaug[f'{arch}_preds'][idx]:
                match_count+=1
                selected_samples_per_model[unlabeled_set[idx]]['preds'].append(best_model_cycle_preds_wkaug[f'{arch}_preds'][idx])
                selected_samples_per_model[unlabeled_set[idx]]['confs'].append(best_model_cycle_preds_wkaug[f'{arch}_confs'][idx])
            else:
                mismatch_count+=1
                mismatch_set[f'{arch}'].append([unlabeled_set[idx], 
                                                best_model_cycle_preds[f'{arch}_preds'][idx],
                                                best_model_cycle_preds_wkaug[f'{arch}_preds'][idx],
                                                best_model_cycle_preds[f'{arch}_confs'][idx],
                                                best_model_cycle_preds_wkaug[f'{arch}_confs'][idx]])
    
    do_random = False
    common_samples = defaultdict(lambda: defaultdict(list))
    for i in selected_samples_per_model.keys():
        if do_random:
            if all(value > 0.9 for value in selected_samples_per_model[i]['confs']):
                common_samples[i]['preds'] = selected_samples_per_model[i]['preds']
                common_samples[i]['confs'] = selected_samples_per_model[i]['confs']
        
        #else select the most agreed upon samples
        elif len(selected_samples_per_model[i]['preds'])==5:
            common_samples[i]['preds'] = selected_samples_per_model[i]['preds']
            common_samples[i]['confs'] = selected_samples_per_model[i]['confs']
    
    selected_indexes = defaultdict()
    label_counter = defaultdict(int)
    to_select = 15000
    for i in common_samples.keys():
        if label_counter[common_samples[i]['preds'][0]]>=100:
            continue
        if len(set(common_samples[i]['preds']))==1:
    #     if random.random() < to_select/len(common_samples.keys()):
            label_counter[common_samples[i]['preds'][0]]+=1
            selected_indexes[i] = common_samples[i]['preds'][0]
            
    for arch in arch_list:        
        print(arch)
        model = CommitteeMember(arch,n_classes)
        thief_model = model.model
        testset_pretrained = load_victim_dataset(cfg.VICTIM.DATASET,arch,load_for_pretrained=True,
                                                pretrained_transform=model.transforms)
        test_loader_pretrained = DataLoader(testset_pretrained, batch_size=cfg.TRAIN.BATCH, num_workers=4, shuffle=False,
                                            pin_memory=False)

        #load data for training the thief model
        
        thief_data = load_thief_dataset(cfg.THIEF.DATASET,cfg.THIEF.DATA_ROOT,model.transforms)
        strong_aug = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        # RandAugmentMC(n=2,m=10),
                                        model.transforms])
        thief_data_strong_aug = load_thief_dataset(cfg.THIEF.DATASET,cfg.THIEF.DATA_ROOT, strong_aug)
        
        cycle = best_model_cycle[arch]
        print(f"Best cycle for model {arch} is {cycle}")
        with open(os.path.join(cfg.SAVE_DIR,"labeled_set_trial_1_cycle_10"),'rb') as f:
            labeled_set = pickle.load(f)
        with open(os.path.join(cfg.SAVE_DIR,"val_set_trial_1_cycle_10"),'rb') as f:
                val_set = pickle.load(f)
        print(f"Num labeled samples: {len(labeled_set)}")
        
        pred_save_path = os.path.join(cfg.SAVE_DIR,f'trial_1_cycle_{cycle}_{arch}_unlabeled_pred.dict')
        with open(pred_save_path,'rb') as f:
            cycle_preds = pickle.load(f)
        if cycle!=10:
            for i in tqdm(list(cycle_preds.keys())):
                try:
                    labeled_set.remove(i)
                except:
                    continue
        ssl_indexes = list(selected_indexes.keys())
        
        #In fixmatch, Unlabeled set will be used for training separately, with strong augmentation inputs and computed weak
        #augmentation pseudo labels
        unlabeled_set = ssl_indexes
        print(f"after SSL addition, U_L:{len(labeled_set)}, U_UL:{len(unlabeled_set)}")
        
        # Create train, val and unlabeled dataloaders
        train_loader, val_loader, unlabeled_loader = create_thief_loaders_ssl_fixmatch(thief_data, query_data, thief_data_strong_aug, 
                                                                            target_model, labeled_set,
                                                                            val_set, unlabeled_set, 
                                                                            128, selected_indexes)
    
        dataloaders  = {'train': train_loader, 'test': test_loader_pretrained, 'val': val_loader, 'unlabeled': unlabeled_loader}
        print("Dataloaders created")

        print(f"Best cycle for model {arch} is {cycle}")
        save_path = os.path.join(cfg.SAVE_DIR,f'trial_1_cycle_{cycle}_{arch}_best.pth')
        state_dict = torch.load(save_path)['state_dict']
        # Load thief model
        thief_model.load_state_dict(state_dict)
        acc_test, _ = testz(thief_model, test_loader_pretrained)
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.SGD(thief_model.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
        save_path = os.path.join(cfg.SAVE_DIR,f'{arch}_{best_model_cycle[arch]}_ssl_Fixmatch.pth')
        print("Training new Model")
        total_loss = 0

        train_with_validation_fixmatch(thief_model, criterion, optimizer, scheduler, dataloaders, 100, 1, 0,save_path)
        acc, f1 = testz(thief_model, test_loader_pretrained)
        print(f'Initial model on target dataset: acc = {acc:.4f}, f1 = {f1:.4f}')
        gc.collect()
        torch.cuda.empty_cache()
        
    final_preds_all = defaultdict(list)
    for img,label in tqdm(test_loader):
        img = img.cuda()
        pred = target_model(img).argmax(axis=1,keepdim=False)
        pred = pred.detach().cpu().tolist()
        label=label.detach().cpu().tolist()
        final_preds_all['gt'].extend(label)
        final_preds_all['victim'].extend(pred)
        
    for arch in arch_list:
        save_path_ssl  = os.path.join(cfg.SAVE_DIR,f'{arch}_{best_model_cycle[arch]}_ssl_Fixmatch.pth')
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
        final_pred = counts.most_common(1)[0][0]
        if counts[final_pred]<2:
            final_pred=-1
        combined_preds.append(final_pred)
    acc = accuracy_score(final_preds_all['gt'],combined_preds)
    agr = accuracy_score(final_preds_all['victim'],combined_preds)
    
    print("Ensemble accuracy")
    print(acc,agr)
            
    import pdb;pdb.set_trace()