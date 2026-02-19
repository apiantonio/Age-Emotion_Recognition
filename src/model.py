import torch
import torch.nn as nn
from torchvision import models

class FaceModel(nn.Module):
    def __init__(self, model_name='resnet50', mode='classification', num_classes=7, pretrained=True, freeze_percentage=0.0):
        super(FaceModel, self).__init__()
        
        self.model_name = model_name
        print(f"Loading {model_name} (Pretrained={pretrained})...")
        
        if model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features # 2048
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features # 512
            self.backbone.fc = nn.Identity()
        
        elif model_name == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            in_features = self.backbone.fc.in_features # 512
            self.backbone.fc = nn.Identity()
        
        elif model_name == 'resnet101':
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            in_features = self.backbone.fc.in_features # 2048
            self.backbone.fc = nn.Identity()
        
        elif model_name == 'resnet152':
            weights = models.ResNet152_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet152(weights=weights)
            in_features = self.backbone.fc.in_features # 2048
            self.backbone.fc = nn.Identity()
                   
        elif model_name == 'efficientnet_b0':
            # 🏆 CONSIGLIATO PER WEBCAM
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features # 1280
            # EfficientNet ha una struttura diversa (classifier invece di fc)
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'efficientnet_b2': 
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b2(weights=weights)
            # B2 ha un feature vector di 1408
            in_features = self.backbone.classifier[1].in_features 
            self.backbone.classifier = nn.Identity()
        
        elif model_name == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            # B3 ha un feature vector di 1536
            in_features = self.backbone.classifier[1].in_features 
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'mobilenet_v3_large':
            # CONSIGLIATO PER CPU PURE
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            in_features = self.backbone.classifier[0].in_features # 960
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'convnext_tiny':
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            in_features = self.backbone.classifier[2].in_features # 768
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'efficientnet_v2_s': 
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            in_features = self.backbone.classifier[1].in_features 
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Modello {model_name} non supportato.")


        if freeze_percentage > 0:
            # Funziona genericamente su tutti i modelli iterando sui parametri
            total_params = len(list(self.backbone.parameters()))
            params_to_freeze = int(total_params * freeze_percentage)
            for i, param in enumerate(self.backbone.parameters()):
                if i < params_to_freeze:
                    param.requires_grad = False

        # Adattiamo la dimensione hidden in base alla backbone
        hidden_dim = in_features // 2 
        
        if mode == 'classification':
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.6),
                nn.Linear(hidden_dim, num_classes)
            )
        elif mode == 'regression':
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.6),
                nn.Linear(hidden_dim, 1)
            )

        # Inizializzazione pesi della head
        self._init_weights(self.head)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        
        # Se il tensore è 4D (es. ConvNeXt: Batch, C, 1, 1), lo appiattiamo
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
            
        output = self.head(features)
        return output
