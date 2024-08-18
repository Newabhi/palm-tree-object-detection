from pathlib import Path
from palmTreesCounter.definitions.config_entity import BaseModelConfig
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn


class BaseModel:
    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_base_model(self):
        # Use the provided get_model function to define the model
        self.model = self.get_model(self.config.params_classes)
        self.model = self.model.to(self.device)
        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def get_model(num_classes):
        # Load the pre-trained FasterR-CNN model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Modify the head of the model to match the number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        return model
    
    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)

