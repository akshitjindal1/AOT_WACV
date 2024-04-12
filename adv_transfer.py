from aot.aot import *
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