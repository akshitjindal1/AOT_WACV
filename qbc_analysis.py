from qbc import *
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
    # indices = list(range(min(1281167, len(query_data))))
    torch.manual_seed(43)
    random.seed(43)
    n_cycles = cfg.ACTIVE.CYCLES
    with open(os.path.join(cfg.SAVE_DIR,f"labeled_set_trial_1_cycle_{n_cycles}"),'rb') as f:
        labeled_set = pickle.load(f)
    with open(os.path.join(cfg.SAVE_DIR,f"val_set_trial_1_cycle_{n_cycles}"),'rb') as f:
            val_set = pickle.load(f)
    with open(os.path.join(cfg.SAVE_DIR,f"unlabeled_set_trial_1_cycle_{n_cycles}"),'rb') as f:
            unlabeled_set = pickle.load(f)
            
    arch_list = cfg.THIEF.ARCH
    print(arch_list)
    
    #find the best cycle for each ensemble member
    dic = defaultdict(list)
    best_model_cycle_preds = defaultdict(list)
    best_model_cycle = defaultdict()
    for arch in arch_list:
        model = CommitteeMember(arch,n_classes)
        thief_model = model.model
        testset_pretrained = load_victim_dataset(cfg.VICTIM.DATASET,arch,load_for_pretrained=True,pretrained_transform=model.transforms)
        test_loader_pretrained = DataLoader(testset_pretrained, batch_size=128, num_workers=4, shuffle=False, pin_memory=False)

        #load data for training the thief model
        thief_data = load_thief_dataset(cfg.THIEF.DATASET,cfg.THIEF.DATA_ROOT,model.transforms)
        #data used to query the target model has to be without augmentations
    #     print(f"Num labeled samples: {len(labeled_set)}")
    #     print(f"Num unlabeled samples: {len(unlabeled_set)}")

        # Create train, val and unlabeled dataloaders
        train_loader, val_loader, unlabeled_loader = create_thief_loaders(thief_data, query_data, target_model, labeled_set, val_set, unlabeled_set, 128)
        # train_loader_noaug, val_loader_noaug, unlabeled_loader_noaug = create_thief_loaders(query_data, query_data,target_model, labeled_set, val_set, unlabeled_set,128)
        # dataloaders  = {'train': train_loader, 'test': test_loader_pretrained, 'val': val_loader, 'unlabeled': unlabeled_loader}
        # dataloaders_noaug = {'train': train_loader_noaug, 'val': val_loader_noaug, 'unlabeled': unlabeled_loader_noaug}
    #     print("Dataloaders created")
        
        #find the cycle number with best validation accuracy for each committee member
        best_acc = 0.0
        for cycle in range(1,n_cycles+1):
            print(f"cycle {cycle} stats")
            save_path = os.path.join(cfg.SAVE_DIR,f'trial_1_cycle_{cycle}_{arch}_best.pth')
            state_dict = torch.load(save_path)['state_dict']
            # Load thief model
            thief_model.load_state_dict(state_dict)
            # Compute metrics on test dataset
            acc, _ = testz(thief_model, val_loader)
            acc_test, _ = testz(thief_model, test_loader_pretrained)
            dic[cycle].append(acc_test)
            print(f'cycle {cycle} model acc on val set {acc:.4f}, test set = {acc_test:.4f}')
            if acc_test>best_acc:
                best_model_cycle[arch] = cycle
                best_acc = acc_test
            
        #test_loader contains the original data labels
        #agreement will be computed later but need to save the predictions of test_loader by target_model
        cycle = best_model_cycle[arch]
        print(f"Best cycle for model {arch} is {cycle}")
        save_path = os.path.join(cfg.SAVE_DIR,f'trial_1_cycle_{cycle}_{arch}_best.pth')
        state_dict = torch.load(save_path)['state_dict']
        # Load thief model
        thief_model.load_state_dict(state_dict)
        for img, label in test_loader_pretrained:
        # for img, label,_ in test_loader_pretrained: #FOR CALTECH256 and other datasets
            img = img.cuda()
            preds = thief_model(img)
            preds = list(preds.cpu().detach().numpy())
            pred_labels = []
            for pred in preds:
                pred = np.argmax(softmax(pred))
                pred_labels.append(pred)
            best_model_cycle_preds[arch].extend(pred_labels)

    for img, label in test_loader:
    # for img, label,_ in test_loader: #FOR CALTECH256 and other datasets
        img = img.cuda()
        label = list(label.cpu().detach().numpy())
        preds = target_model(img).argmax(axis=-1,keepdims=False)
        preds = list(preds.cpu().detach().numpy())
        best_model_cycle_preds['victim'].extend(preds)
        best_model_cycle_preds['gt'].extend(label)
    
    model_vs_cycleacc = defaultdict(list)
    for i in range(1,11):
        for j,arch in enumerate(arch_list):
            model_vs_cycleacc[arch].append(float(f"{dic[i][j]*100:.2f}"))
    
    fig = px.line(x = list(dic.keys()), y = list(model_vs_cycleacc.values()),markers=True,
              title="Committee accuracy per cycle")
    fig.update_xaxes(title_text='Cycle')
    fig.update_yaxes(title_text='Accuracy')
    for i,f in enumerate(fig.data):
        f.name = arch_list[i]
        fig.add_scatter(x = [f.x[-1]], y = [f.y[-1]],
                    mode = 'markers + text',
                        marker = {'color':f['line']['color'], 'size':10},
                        showlegend = False,
                        text = [f.y[-1]],
                        textposition='middle right')
    fig.update_layout(legend_title_text='Model Arch.')

    pio.write_image(fig, os.path.join(cfg.SAVE_DIR,'training_cycles.pdf'))
    
    dic_df = {'MODEL':[],'ACC':[],'AGR':[]}
    for arch in arch_list:
        dic_df['MODEL'].append(arch)
        acc,agr = acc_and_agr(best_model_cycle_preds[arch],best_model_cycle_preds['gt'],best_model_cycle_preds['victim'])
        dic_df['ACC'].append(acc)
        dic_df['AGR'].append(agr)
    
    print(dic_df)

    combined_preds = []
    for i in tqdm(range(len(testset))):
        all_model_preds_i = []
        for arch in arch_list:
            all_model_preds_i.append(best_model_cycle_preds[arch][i])
        counts = Counter(all_model_preds_i)
        final_pred = counts.most_common(1)[0][0]
        if counts[final_pred]<2:
            final_pred=-1
        combined_preds.append(final_pred)
    
    acc,agr = acc_and_agr(combined_preds,best_model_cycle_preds['gt'],best_model_cycle_preds['victim'])
    print(f"Ensemble accuracy:{acc} agreement:{agr}")
    
    with open(os.path.join(cfg.SAVE_DIR, "best_model_cycle_dic.pkl"), 'wb') as f:
        pickle.dump(best_model_cycle,f)
    
    
    import pdb;pdb.set_trace()