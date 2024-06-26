import pandas as pd
import os
import requests
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
from io import BytesIO


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, labels_col, urls_col, transform=None):
        print("Initializng Dataset with file path :::",csv_file)
        self.annotations = pd.read_csv(csv_file)
        self.labels_col = labels_col
        self.urls_col = urls_col
        self.transform = transform

    def __len__(self):
        return len(self.annotations) - 1

    def __getitem__(self, index):
#         try:
#         print("DEBUGGING GET ITEM:::",self.annotations,len(self.annotations))
        label = self.annotations.iloc[index, self.labels_col]
        img_url = self.annotations.iloc[index,self.urls_col]

        image = download_preprocess_image(img_url)
        if self.transform:
            image = self.transform(image)
#         except Exception as e:
#             print(len(self.annotations),type(self.annotations),self.annotations.columns)
            
        return image, label

def get_data_loader(csv_file, labels_col, urls_col, batch_size, transform):
    dataset = CustomImageDataset(csv_file=csv_file, labels_col=labels_col, urls_col=urls_col,
                                 transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def download_preprocess_image(url):
    image = None
    try:
        response = requests.get(url, timeout=5)  # Set timeout as per your requirement
        response.raise_for_status()  # This will raise an exception for HTTP errors
         # If the request was successful, proceed with processing the image
        image = Image.open(BytesIO(response.content))
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print("Retrying")# Handle HTTP errors like 502

        time.sleep(10)
        response = requests.get(url, timeout=5)  # Set timeout as per your requirement
        response.raise_for_status()  # This will raise an exception for HTTP errors
         # If the request was successful, proceed with processing the image
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as err:
        print(f"An error occurred: {err}")  # Handle other errors like timeouts
        return None
    return image
