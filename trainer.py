import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd 
import wandb
from tqdm import tqdm
import time 

class Trainer:
    def __init__(self, model, dataloader, test_dataloader, epochs, learning_rate, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.epochs = epochs
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.test_dataloader = test_dataloader
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, labels in tqdm(self.dataloader):
            inputs, labels = inputs.to(self.device), labels.type(torch.LongTensor).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader)
        return avg_loss

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            avg_loss = self.train_epoch()
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")
            wandb.log({"Epoch": epoch, "Loss": avg_loss})
            _, _ = self.evaluate()

    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_preds = []
        print("Evaluating")
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_dataloader):
                inputs, labels = inputs.to(self.device), labels.type(torch.LongTensor).to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        print(len(preds),len(labels))
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
#         auc = roc_auc_score(all_labels, all_preds, average='weighted')

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        wandb.log({"Test Accuracy": accuracy, "Test Precision": precision, "Test Recall": recall, "Test F1 Score": f1})
        return all_labels, all_preds
 
    def generate_predictions_csv(self,csv_file,test_output_file):
        #todo: Combine this script and inference script into one
        '''
        Generates a csv file containing the outputs  on the test csv
        csv_file : input csv file (train or test at the moment)
        test_output_file: output file name
        '''
        
        labels, preds = self.evaluate()
        annotations = pd.read_csv(csv_file)

        annotations['predictions'] = preds 
        annotations['labels'] = labels 

        annotations.to_csv(test_output_file,index=False)
        print(f"Outputs saved at {test_output_file}")



    def inference(self, csv_file, dataloader):
        '''
        Runs inference on the csv and attaches labels to it
        '''
        st = time.time()
        data = pd.read_csv(csv_file)
        
        self.model.eval()
        all_preds = list()
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())

        data['predictions'] = all_preds
        data.to_csv(csv_file,index=False)
        print(f"Inference completed in {time.time() - st}s")

