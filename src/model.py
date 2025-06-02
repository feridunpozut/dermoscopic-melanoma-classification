'''
Model definition for a lightweight skin lesion classifier using EfficientNet.
'''
import torch.nn as nn
import timm

class LiteSkinLesionClassifier(nn.Module):
    """
    A lightweight skin lesion classifier based on EfficientNet.
    This model is designed for binary classification of skin lesions into benign and malignant categories.
    It uses a pre-trained EfficientNet backbone and adds a dropout layer followed by a linear classifier.
    Args:
        model_name (str): The name of the EfficientNet model to use (default: 'efficientnet_b0').
        pretrained (bool): Whether to use a pre-trained model (default: True).
        dropout_rate (float): Dropout rate for the classifier layer (default: 0.3).
    """
    def __init__(self, model_name='efficientnet_b0', pretrained=True, dropout_rate=0.3):
        """
        Initializes the LiteSkinLesionClassifier.
        Args:
            model_name (str): The name of the EfficientNet model to use.
            pretrained (bool): Whether to use a pre-trained model.
            dropout_rate (float): Dropout rate for the classifier layer.
        """
        super(LiteSkinLesionClassifier, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2) with class scores.
        """
        features = self.backbone(x)
        out = self.classifier(features)
        return out
