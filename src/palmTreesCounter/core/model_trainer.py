from pathlib import Path
from palmTreesCounter import logger
import mlflow
from palmTreesCounter.definitions.config_entity import TrainingConfig
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from palmTreesCounter.definitions.palm_tree_dataset import PalmTreeDataset
import os
import pandas as pd
from tqdm import tqdm
from palmTreesCounter.utils.metrics import compute_metrics



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_base_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        num_classes = self.config.params_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Load the pretrained model
        model.load_state_dict(torch.load(self.config.base_model_path, map_location=self.device, weights_only=True))
        
        self.model = model
    

    def get_transforms(self):
        return T.Compose([
            T.ToTensor(),                     
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_data_loaders(self):
        transforms = self.get_transforms()
        train_df = pd.read_csv(os.path.join(self.config.training_data, "train_labels.csv"))
        test_df = pd.read_csv(os.path.join(self.config.training_data, "test_labels.csv"))
        train_img_dir = os.path.join(self.config.training_data, "train") 
        test_img_dir = os.path.join(self.config.training_data, "test") 

        # Load the datasets
        train_dataset = PalmTreeDataset(train_df, train_img_dir, transforms=transforms, target_size=(self.config.params_image_size, self.config.params_image_size))
        test_dataset = PalmTreeDataset(test_df, test_img_dir, transforms=transforms, train=False, target_size=(self.config.params_image_size, self.config.params_image_size))
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        test_loader = DataLoader(test_dataset, batch_size=self.config.params_batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        
        # Save the data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader



    def train(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_experiment('palm_tree_counting')
        
        with mlflow.start_run(run_name="Training"):
            self.model.to(self.device)

            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        
            for epoch in range(self.config.params_epochs):
                self.model.train()
                train_loss = 0

                for imgs, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.params_epochs}", leave=False):
                    imgs = [img.to(self.device) for img in imgs]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    loss_dict = self.model(imgs, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    train_loss += losses.item()
                    
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                lr_scheduler.step()
                avg_train_loss = train_loss / len(self.train_loader)
                logger.info(f'Epoch {epoch + 1} - Training Loss: {avg_train_loss}')

                # Evaluate the model on the test set (used as validation)
                self.validate_model()
                
                mlflow.log_metrics({'train_loss': avg_train_loss}, step=epoch+1)
                mlflow.log_params(self.config.all_params)

            mlflow.pytorch.log_model(self.model, "model", registered_model_name="PalmTreeCounter (FastRCNNPredictor)")
            self.save_model(self.model, self.config.trained_model_path)


    def validate_model(self):
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, targets in tqdm(self.test_loader, desc="Validation", leave=False):
                imgs = [img.to(self.device) for img in imgs]
                targets = [t.to(self.device) for t in targets]
                
                # Get the predictions from the model
                predictions = self.model(imgs)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Compute evaluation metrics (MAE and RMSE)
        mae, rmse = compute_metrics(all_predictions, all_targets)
        
        logger.info(f'Mean Absolute Error: {mae}')
        logger.info(f'Root Mean Squared Error: {rmse}')
        
        mlflow.log_metrics({'mae': mae, 'rmse': rmse})


    @staticmethod
    def save_model(model: nn.Module, path: Path):
        torch.save(model.state_dict(), path)

    
