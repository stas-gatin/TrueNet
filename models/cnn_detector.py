# models/cnn_detector.py
import torch
import torch.nn as nn
import torchvision.models as models

class HybridRealFakeCNN(nn.Module):
    """
    Hybrid Convolutional Neural Network processing both image features and handcrafted features.
    """
    def __init__(self, backbone_name='resnet18', pretrained=True, hand_feat_dim=14, freeze_until=0.8):
        super().__init__()

        if not hasattr(models, backbone_name):
            raise ValueError(f"Unknown backbone '{backbone_name}' in torchvision.models")
        backbone_fn = getattr(models, backbone_name)
        backbone = backbone_fn(pretrained=pretrained)

        # Retain CNN features without the classification head
        self.cnn_features = nn.Sequential(*list(backbone.children())[:-1])  # [B, C, 1, 1]

        # Determine output feature dimensions dynamically
        if hasattr(backbone, 'classifier'):  # EfficientNet
            if isinstance(backbone.classifier, nn.Linear):
                cnn_out_dim = backbone.classifier.in_features
            elif isinstance(backbone.classifier, nn.Sequential):
                for layer in reversed(backbone.classifier):
                    if isinstance(layer, nn.Linear):
                        cnn_out_dim = layer.in_features
                        break
        elif hasattr(backbone, 'fc'):  # ResNet
            cnn_out_dim = backbone.fc.in_features
        else:
            raise ValueError("Cannot determine output feature size of backbone")

        # Freeze layers (optional)
        for param in list(self.cnn_features.parameters())[:-4]:
            param.requires_grad = False

        # Hybrid classifier block
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(cnn_out_dim + hand_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x, hand_features):
        """Forward pass for the hybrid network."""
        cnn_out = self.cnn_features(x)                   # [B, C, 1, 1]
        cnn_out = cnn_out.view(cnn_out.size(0), -1)      # [B, C]
        combined = torch.cat([cnn_out, hand_features], dim=1)
        return self.classifier(combined)


class ImprovedHybridCNN(nn.Module):
    """
    Improved Hybrid CNN model with an additional Multi-layer Perceptron (MLP) for handcrafted features.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True, hand_feat_dim=14, hidden_dim=1024, dropout_prob=0.4, freeze_until=0.8):
        super().__init__()

        # Backbone
        if not hasattr(models, backbone_name):
            raise ValueError(f"Unknown backbone '{backbone_name}' in torchvision.models")
        backbone_fn = getattr(models, backbone_name)
        backbone = backbone_fn(pretrained=pretrained)

        # Retain CNN features without the classification head
        self.cnn_features = nn.Sequential(*list(backbone.children())[:-1])  # [B, C, 1, 1]
        cnn_out_dim = list(backbone.children())[-1].in_features

        # Freeze early layers (optional)
        for param in list(self.cnn_features.parameters())[:-4]:
            param.requires_grad = False

        # MLP for handcrafted features
        self.hand_feat_mlp = nn.Sequential(
            nn.Linear(hand_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1)
        )

        # Hybrid classifier block
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(cnn_out_dim + 128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, hand_features):
        """Forward pass for improved hybrid architecture."""
        cnn_out = self.cnn_features(x)                     # [B, C, 1, 1]
        cnn_out = cnn_out.view(cnn_out.size(0), -1)       # [B, C]
        hand_out = self.hand_feat_mlp(hand_features)      # [B, 128]
        combined = torch.cat([cnn_out, hand_out], dim=1)
        return self.classifier(combined)


class HybridRealFakeCNN_Normalized(nn.Module): 
    """
    Hybrid CNN Model with Layer Normalization on the concatenated feature vector.
    Pending testing phase.
    """
    def __init__(self, backbone_name='efficientnet_b3', pretrained=True, hand_feat_dim=14, freeze_until=0.8):
        super().__init__()

        if not hasattr(models, backbone_name):
            raise ValueError(f"Unknown backbone '{backbone_name}'")
        
        backbone_fn = getattr(models, backbone_name)
        # Use 'weights'='IMAGENET1K_V1' for EfficientNet models
        backbone = backbone_fn(weights='IMAGENET1K_V1' if pretrained else None)

        # Extract only CNN features without the classification head
        self.cnn_features = nn.Sequential(*list(backbone.children())[:-1]) 
        
        # Determine the correct output dimension (EfficientNetB3 expects 1536)
        if 'efficientnet' in backbone_name:
            if backbone_name == 'efficientnet_b3':
                 cnn_out_dim = 1536
            else:
                 # Pytorch EfficientNet logic extraction
                 cnn_out_dim = backbone.features[-1][0].out_channels
        else:
             cnn_out_dim = list(backbone.children())[-1].in_features 

        # --- Freeze layers (similar to original approach) ---
        for param in list(self.cnn_features.parameters())[:-4]:
            param.requires_grad = False
        
        # --- L2 Normalization representation mapping  ---
        self.l2_norm = nn.LayerNorm(cnn_out_dim + hand_feat_dim)

        # --- Hybrid Classifier Block ---
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + hand_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x, hand_features):
        """Forward evaluation execution process logic."""
        cnn_out = self.cnn_features(x)         # [B, C, 1, 1]
        cnn_out = cnn_out.view(cnn_out.size(0), -1)   # [B, C]
        
        # 1. Feature concatenation combinations
        combined = torch.cat([cnn_out, hand_features], dim=1)
        
        # 2. Normalize the concatenated features
        normalized_combined = self.l2_norm(combined)
        
        # 3. Predict inference probabilities
        return self.classifier(normalized_combined)


class HybridRealFakeCNN_HeavyDrop(nn.Module):
    """
    Hybrid CNN with a severely heavy dropout rate integration layer structure.
    """
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True, hand_feat_dim=14, freeze_until=0.8):
        super().__init__()

        if not hasattr(models, backbone_name):
            raise ValueError(f"Unknown backbone '{backbone_name}'")
        
        backbone_fn = getattr(models, backbone_name)
        backbone = backbone_fn(weights='IMAGENET1K_V1' if pretrained else None)

        # Splice classifier outputs
        self.cnn_features = nn.Sequential(*list(backbone.children())[:-1])

        # Evaluate dimension representation
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)  # Image matching layout mapped shapes
            out = self.cnn_features(dummy)
            cnn_out_dim = out.view(1, -1).size(1)

        # Freeze mapping sizes layers limits targets
        for param in list(self.cnn_features.parameters())[:-4]:
            param.requires_grad = False

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(cnn_out_dim + hand_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 2)
        )

    def forward(self, x, hand_features):
        """Forward network step configuration evaluation logic"""
        cnn_out = self.cnn_features(x)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        combined = torch.cat([cnn_out, hand_features], dim=1)
        return self.classifier(combined)
  

class AugmentedFeatureFusionNet(nn.Module): 
    """
    Hybrid network combining EfficientNet block with Layer Normalization for feature fusion
    and a heavily optimized classification head map.
    """
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True, hand_feat_dim=14, freeze_until=0.8):
        super().__init__()
        
        # --- 1. CNN Backbone Setup ---
        if not hasattr(models, backbone_name):
            raise ValueError(f"Unknown backbone '{backbone_name}'")
            
        backbone_fn = getattr(models, backbone_name)
        # Using 'weights' for better mapping structures
        backbone = backbone_fn(weights='IMAGENET1K_V1' if pretrained else None)

        # Load pure features elements
        self.cnn_features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # EfficientNet-B0 => cnn_out_dim = 1280
        # EfficientNet-B3 => cnn_out_dim = 1536
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.cnn_features(dummy)
            out = self.avgpool(out)
            cnn_out_dim = out.view(1, -1).size(1)

        # --- Freezing mappings elements blocks (Fine-tuning) ---
        # Fine-tune the last layers definitions elements
        total_features = len(list(self.cnn_features.parameters()))
        freeze_until = int(total_features * freeze_until) 

        for i, param in enumerate(self.cnn_features.parameters()):
            if i < freeze_until:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # --- 2. Feature Fusion Block ---
        total_input_dim = cnn_out_dim + hand_feat_dim
        bottleneck_dim = 512 

        # Layer Normalization
        self.feature_norm = nn.LayerNorm(total_input_dim) 

        # --- 3. Optimized Classifier (2-layer MLP) ---
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            # Bottleneck layer reduces vectors
            nn.Linear(total_input_dim, bottleneck_dim), 
            nn.ReLU(),
            # Layer 1
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.6), # Elevated dropout for regularization
            # Output layer
            nn.Linear(256, 2)
        )

    def forward(self, x, hand_features):
        """Forward step execution."""
        # 1. Fetch CNN values
        cnn_out = self.cnn_features(x)
        cnn_out = self.avgpool(cnn_out)        # [B, 1536, 1, 1]
        
        cnn_out = cnn_out.view(cnn_out.size(0), -1) # [B, 1536]
        
        # 2. Integrate hand-crafted features
        combined = torch.cat([cnn_out, hand_features], dim=1)
        normalized_combined = self.feature_norm(combined)
        
        # 3. Classify the combined features
        return self.classifier(normalized_combined)
    

class FrequencyAwareAFFNet(nn.Module):
    """
    State of the art sequence evaluation Hybrid CNN based on EfficientNet with cross-attention.
    """
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True, hand_feat_dim=14, freeze_until=0.8, dropout_rate=0.6):
        super().__init__()
        
        # --- 1. CNN Backbone ---
        if not hasattr(models, backbone_name):
            raise ValueError(f"Unknown backbone '{backbone_name}'")
        
        backbone_fn = getattr(models, backbone_name)
        backbone = backbone_fn(weights='IMAGENET1K_V1' if pretrained else None)

        self.cnn_features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Extract shapes dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.cnn_features(dummy)
            out = self.avgpool(out)
            cnn_out_dim = out.view(1, -1).size(1)

        # --- Freezing layout boundaries (80% frozen targets) ---
        total_features = len(list(self.cnn_features.parameters()))
        freeze_until = int(total_features * freeze_until)

        for i, param in enumerate(self.cnn_features.parameters()):
            if i < freeze_until:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # --- 2. Frequency-Aware Attention Block ---
        self.cnn_out_dim = cnn_out_dim
        
        # Attention Map Generator
        self.attention_generator = nn.Sequential(
            nn.Linear(hand_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.cnn_out_dim), # Mapped layouts boundaries
            nn.Sigmoid() # Restricting to ranges
        )

        # --- 3. Final Fusion settings ---
        total_fused_dim = self.cnn_out_dim + hand_feat_dim
        bottleneck_dim = 512

        self.feature_norm = nn.LayerNorm(total_fused_dim) 

        self.classifier = nn.Sequential(
            nn.Linear(total_fused_dim, bottleneck_dim), 
            nn.ReLU(),
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(256, 2)
        )

    def forward(self, x, hand_features):
        """Evaluation pass."""
        # 1. Fetch parameters
        cnn_out = self.cnn_features(x)
        cnn_out = self.avgpool(cnn_out)        
        Z_CNN = cnn_out.view(cnn_out.size(0), -1) # [B, C_CNN]

        # 2. Emit boundaries dictionaries
        attention_weights = self.attention_generator(hand_features) # [B, C_CNN]
        
        # 3. Multiply references templates
        Z_CALIBRATED = Z_CNN * attention_weights # Hadamard product
        
        # 4. Integrate representations
        combined = torch.cat([Z_CALIBRATED, hand_features], dim=1)
        normalized_combined = self.feature_norm(combined)
        
        # 5. Extract values
        return self.classifier(normalized_combined)