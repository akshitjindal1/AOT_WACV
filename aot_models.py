import torch
import torchvision
from models.cifar10_models import resnet34 as cifar10_resnet34
import torchvision.transforms as transforms

class CommitteeMember:
    def __init__(self,arch,n_classes):
        self.model_arch = arch
        self.model_weights = self.load_thief_model_weights(self.model_arch)
        self.model = self.load_thief_model(self.model_arch,n_classes,self.model_weights)
        if arch!='resnet32':
            self.transforms = self.model_weights.transforms()
        else:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
            ])

    def load_thief_model_weights(self,arch):
        if arch=="resnet32":
            weights = None

        elif arch == "resnet34":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="ResNet34_Weights.IMAGENET1K_V1")

        elif arch=="alexnet":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="AlexNet_Weights.IMAGENET1K_V1")
        
        elif arch=="mobilenet_v3_large":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="MobileNet_V3_Large_Weights.IMAGENET1K_V2")
        
        elif arch=="densenet121":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="DenseNet121_Weights.IMAGENET1K_V1")
        
        elif arch=="efficientnet_v2_s":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_V2_S_Weights.IMAGENET1K_V1")

        elif arch=="efficientnet_b0":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B0_Weights.IMAGENET1K_V1")

        elif arch=="efficientnet_b1_v1":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B1_Weights.IMAGENET1K_V1")
        elif arch=="efficientnet_b1_v2":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B1_Weights.IMAGENET1K_V2")


        elif arch=="efficientnet_b2":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B2_Weights.IMAGENET1K_V1")
        
        elif arch=="efficientnet_b3":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B3_Weights.IMAGENET1K_V1")

        elif arch=="efficientnet_b6":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B6_Weights.IMAGENET1K_V1")
        elif arch=="efficientnet_b7":
            weights = torch.hub.load("pytorch/vision:v0.14.1", "get_weight", name="EfficientNet_B7_Weights.IMAGENET1K_V1")
        return weights
    

    
    def load_thief_model(self,arch, n_classes,weights):
        if arch=="resnet32":
            thief_model = cifar10_resnet34(num_classes=n_classes)

        elif arch=="resnet34":
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","resnet34", weights=weights)
            thief_model.fc = torch.nn.Linear(512,n_classes)

        elif arch=="alexnet":
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","alexnet", weights=weights)
            thief_model.classifier._modules['6'] = torch.nn.Linear(4096,n_classes)
        
        elif arch=="mobilenet_v3_large":
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","mobilenet_v3_large", weights=weights)
            thief_model.classifier._modules['3'] = torch.nn.Linear(1280,n_classes)
        
        elif arch=="densenet121":
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","densenet121", weights=weights)
            thief_model.classifier = torch.nn.Linear(1024,n_classes)
        
        elif arch=="efficientnet_b0":
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","efficientnet_b0", weights=weights)
            thief_model.classifier._modules['1'] = torch.nn.Linear(1280,n_classes)

        elif arch in ["efficientnet_b1_v1", "efficientnet_b1_v2"]:
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","efficientnet_b1", weights=weights)
            thief_model.classifier._modules['1'] = torch.nn.Linear(1280,n_classes)

        elif arch=="efficientnet_b2":
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","efficientnet_b2", weights=weights)
            thief_model.classifier._modules['1'] = torch.nn.Linear(1408,n_classes)
        
        elif arch=="efficientnet_v2_s":
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","efficientnet_v2_s", weights=weights)
            thief_model.classifier._modules['1'] = torch.nn.Linear(1280,n_classes)
        
        elif arch=="efficientnet_b3":
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","efficientnet_b3", weights=weights)
            # thief_model.classifier._modules['1'] = torch.nn.Linear(2560,n_classes)
            thief_model.classifier._modules['1'] = torch.nn.Linear(thief_model.classifier._modules['1'].in_features,n_classes)

        elif arch=="efficientnet_b6":
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","efficientnet_b6", weights=weights)
            # thief_model.classifier._modules['1'] = torch.nn.Linear(2560,n_classes)
            thief_model.classifier._modules['1'] = torch.nn.Linear(thief_model.classifier._modules['1'].in_features,n_classes)
        
        elif arch=="efficientnet_b7":
            thief_model = torch.hub.load("pytorch/vision:v0.14.1","efficientnet_b7", weights=weights)
            # thief_model.classifier._modules['1'] = torch.nn.Linear(2560,n_classes)
            thief_model.classifier._modules['1'] = torch.nn.Linear(thief_model.classifier._modules['1'].in_features,n_classes)
        
        thief_model = thief_model.cuda()

        return thief_model
        