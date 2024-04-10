import torch.nn as nn
import torchvision.models as models

class CustomModel(nn.Module):
    def __init__(self, architecture, feature_size = None, num_classes=2, pretrained=False, transfer_learning=False):
        super(CustomModel, self).__init__()
        ''' 
        Model Loading Class: Returns a model to use
        '''


        # Dynamically get the model constructor
        if hasattr(models, architecture): # Model constructor here is (resnet50, resnet18, etc)
            model_constructor = getattr(models, architecture)
            self.model = model_constructor(pretrained=pretrained)
        elif architecture == 'custom' and feature_size: # Use Custom Architecture when you're using a pkl file
            print("Using custom arch")
            self.model = nn.Linear(feature_size, num_classes)
        else:
            raise ValueError(f"Model architecture '{architecture}' is not recognized.")
    
        # Replace the classifier/FC layer's value for the given number of classes
        if 'resnet' in architecture:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif 'vgg' in architecture:
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
        elif 'alexnet' in architecture:
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
        else:
            # You might want to handle other architectures or raise an error
            pass

        # Implement transfer learning
        if transfer_learning:
            # Freeze all layers
            print("Enabling Transfer Learning")
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze the last layer
            if 'resnet' in architecture:
                self.model.fc.weight.requires_grad = True
                self.model.fc.bias.requires_grad = True
            elif 'vgg' in architecture or 'alexnet' in architecture:
                self.model.classifier[6].weight.requires_grad = True
                self.model.classifier[6].bias.requires_grad = True
           
        else: # incase of just using a pkl file 
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
        
    def forward(self, x):
        return self.model(x)
