import yaml
import torchvision.transforms as transforms
from customloader import get_data_loader
from model import CustomModel
from trainer import Trainer
import wandb

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_transforms(transform_config):
    
    transform_list = []
    for transform_name, params in transform_config.items():
        transform_method = getattr(transforms, transform_name)
        if isinstance(params, list):
            print("method::",transform_method,"params::",params)
            transform = transform_method(*params)
        elif params is None:  # For transforms that do not require parameters
            transform = transform_method()
        else:
            transform = transform_method(params)

        transform_list.append(transform)

    composed_transforms = transforms.Compose(transform_list)
    return composed_transforms

def main():
    config = load_config('config.yaml')
    print("Loaded Configs")
    train_transform = get_transforms(config['transforms']['train'])
    test_transform = get_transforms(config['transforms']['test'])
#     csv_file, labels_col, urls_col, batch_size, transform
    train_loader = get_data_loader(config['data']['train_csv'], 
                                   config['train']['annotations'],
                                   config['train']['image_url'], 
                                   config['data']['batch_size'],
                                   train_transform,
                                   config['data']['pkl_path'],
                                   config['train']['pkl_index'])

    test_loader = get_data_loader(config['data']['test_csv'],
                                  config['train']['annotations'], 
                                  config['train']['image_url'],
                                  config['data']['batch_size'],
                                  test_transform,
                                  config['data']['pkl_path'],
                                  config['train']['pkl_index'])
    
    print(f"Url Column :{config['train']['image_url']} \
    \n Labels Column :{config['train']['annotations']}")
    model = CustomModel(architecture=config['model']['architecture'], 
                        feature_size=config['model'].get('embedding_size',None),
                        num_classes=config['model']['num_classes'], 
                        pretrained=config['model']['pretrained'],
                       transfer_learning=config['model']['transfer_learning'])
    print("Model Successfully created")
    wandb.init(project=config['logging']['project_name'], entity=config['logging']['user_name'])
    trainer = Trainer(model, train_loader, test_loader, config['train']['epochs'], config['train']['learning_rate'], config['train']['device'])
    trainer.train()

if __name__ == '__main__':
    main()
