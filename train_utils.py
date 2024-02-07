import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch

def testz(model, dataloader):
    model.eval()

    trues = []
    preds = []
    # print("Calculating metrics")
    # import pdb;pdb.set_trace()
    with torch.no_grad():
        for data in (dataloader):
            inputs = data[0].cuda()
            labels = data[1].cuda()

            scores = model(inputs)
            _, pred = torch.max(scores.data, 1)

            preds.append(pred.cpu())
            trues.append(labels.cpu())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
    
    # print('max predicted label: ', preds.max())
    acc = accuracy_score(y_true=trues, y_pred=preds)
    f1 = f1_score(y_true=trues, y_pred=preds, average='macro')
    
    return acc, f1


def agree(target_model, thief_model, test_loader):
    c=0
    l=0
    target_model.eval()
    thief_model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data[0].cuda()
            n=inputs.shape[0]
            x1=target_model(inputs).argmax(axis=-1,keepdims=False)
            x2=thief_model(inputs).argmax(axis=-1,keepdims=False)
            c+=n-int((torch.count_nonzero(x1-x2)).detach().cpu())
            l+=n
            # print(c, l)
    print('Agreement between Copy and source model is ', c/l)
    return c / l

def agreement_diffaug(target_model, thief_model, test_loader_target, test_loader_thief):
    c=0
    l=0
    target_model.eval()
    thief_model.eval()
    target_preds, thief_preds = [],[]
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader_target):
            inputs = data[0].cuda()
            n=inputs.shape[0]
            x1=target_model(inputs).argmax(axis=-1,keepdims=False)
            target_preds.extend(x1)
        for batch_idx, data in enumerate(test_loader_thief):
            inputs = data[0].cuda()
            n=inputs.shape[0]
            x1=thief_model(inputs).argmax(axis=-1,keepdims=False)
            thief_preds.extend(x1)
        for i in zip(target_preds, thief_preds):
            if i[0]==i[1]:
                c+=1
            l+=1        
    # print('Agreement between Copy and source model is ', c/l)
    return c / l


def dist(indices, dataloader):
    "Return label distribution of selected samples" 
    # create dataloader from dataset
    # dl=DataLoader(dz, batch_size=1, sampler=SubsetRandomSampler(indices), pin_memory=False)
    dl = dataloader
    d = {}
    print('Number of samples ', len(indices))
    # iterator = iter(dl)
    labels = []
    # if target_model is not None:
    #     target_model.eval()
    with torch.no_grad():
        for data in (dl):
            label = data[1]
            # if target_model is not None:
            #     label = target_model(img.cuda()).argmax(axis=1,keepdim=False)
            #     labels.append(label.cpu().detach().numpy())
            # else: 
            labels.extend(label.cpu().detach().numpy())
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        d[int(lbl)] = 0
    for label in labels:
        d[int(label)]+=1
    return d

def train_with_validation(model, criterion, optimizer, scheduler, dataloaders, num_epochs, trial, cycle, out_dir, display_every = 10, early_stop_tolerance=10):
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
            # import pdb;pdb.set_trace()
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

