import os
import mlflow
from torch.utils.data import DataLoader
import pandas as pd
from palmTreesCounter.definitions.config_entity import EvaluationConfig
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from palmTreesCounter import logger
from palmTreesCounter.definitions.palm_tree_dataset import PalmTreeDataset
from palmTreesCounter.utils.metrics import compute_metrics
import torchvision.transforms as T

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_eval_transforms(self):
        return T.Compose([
            T.ToTensor(),                     
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    

    @staticmethod
    def load_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        num_classes = self.config.params_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Load the pretrained model
        model.load_state_dict(torch.load(self.config.path_to_model, map_location=self.device))
        return model
    
    def get_test_dataloader(self):
        transforms = self.get_eval_transforms()
        df = pd.read_csv(os.path.join(self.config.training_data, "test_labels.csv"))
        img_dir = os.path.join(self.config.training_data, "test") 

        # Load the datasets
        test_dataset = PalmTreeDataset(df, img_dir, transforms=transforms, train=False, target_size=(self.config.params_image_size, self.config.params_image_size))

        # DataLoaders
        test_loader = DataLoader(test_dataset, batch_size=self.config.params_batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        
        # Save the data loaders
        self.test_loader = test_loader


    def evaluation(self):
        self.model = self.load_model(self)
        self.model.to(self.device)
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, targets in tqdm(self.test_loader, desc="Evaluation", leave=False):
                imgs = [img.to(self.device) for img in imgs]
                targets = [t.to(self.device) for t in targets]
                
                # Get the predictions from the model
                predictions = self.model(imgs)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Compute evaluation metrics
        self.mae, self.rmse = compute_metrics(all_predictions, all_targets)
        
        logger.info(f'Mean Absolute Error: {self.mae}')
        logger.info(f'Root Mean Squared Error: {self.rmse}')

    
    def send_eval_metric_to_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_experiment('palm_tree_counting')

        with mlflow.start_run(run_name="Evaluation"):
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({'mae': self.mae, 'rmse': self.rmse})
            mlflow.pytorch.log_model(self.model, "model", registered_model_name="PalmTreeCounter (FastRCNNPredictor)")