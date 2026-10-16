import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedAudioBackbone(nn.Module):
    """
    1-channel 2d cnn feature extractor, accepts [Batch, 1, freq, time] -> outputs [batch, 1024] acoustic embedding
    """
    def __init__(self):
        super(SharedAudioBackbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        return torch.flatten(x, 1) # turns to [B, 1024]

class ICBHIModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ICBHIModel, self).__init__()

        # audio cnn branch (named 'backbone' to match trained state_dict)
        self.backbone = SharedAudioBackbone()

        # demographic mlp branch (named 'demo_mlp' to match trained state_dict)
        self.demo_mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True)
        )

        # multimodal classifier head (fusion)
        # audio (1024) + demographics (32) = 1056 input features
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, spectrogram, demographics, return_per_channel=False):
        """
        spectrogram: either 4d [b,1,f,t] or 5d [b,8,1,f,t] for hardware interface
        demographics: [b,5]
        """
        if spectrogram.ndim == 5 or (spectrogram.ndim == 4 and spectrogram.shape[1] == 8):
            if spectrogram.ndim == 5:
                batch_size, num_mics, C, F, T = spectrogram.shape
                x = spectrogram.view(batch_size * num_mics, C, F, T)
            else:
                batch_size, num_mics, F, T = spectrogram.shape
                x = spectrogram.view(batch_size * num_mics, 1, F, T)
            
            channel_feats = self.backbone(x)

            demo_feat = self.demo_mlp(demographics)

            # mode A: diagnostics for each stethoscope
            if return_per_channel:
                # repeat demo vector for each channel
                demo_feat_expanded = demo_feat.repeat_interleave(num_mics, dim=0)
                
                # fuse features per channel [b*8, 1056]
                fused_per_ch = torch.cat((channel_feats, demo_feat_expanded), dim=1)
               
                # compute logits per channel [b*8, 4] and reshape to [b, 8, 4]
                logits_per_ch = self.classifier(fused_per_ch)
                return logits_per_ch.view(batch_size, num_mics, 4)
            

            # mode B: aggregated patient diagnosis with max pooling [b, 4]
            channel_feats = channel_feats.view(batch_size, num_mics, 1024)
            audio_feat, _ = torch.max(channel_feats, dim=1) # [b, 1024]

            fused_features = torch.cat((audio_feat, demo_feat), dim=1)
            return self.classifier(fused_features)

        # standard 1-channel dataset training path
        else:
            audio_feat = self.backbone(spectrogram) # [b, 1024]
            demo_feat = self.demo_mlp(demographics) # [b, 32]

            fused_features = torch.cat((audio_feat, demo_feat), dim=1)
            return self.classifier(fused_features)