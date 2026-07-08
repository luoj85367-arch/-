"""
IMUFrameNet v2 模型定义（自包含，无外部依赖）。

架构：残差 TCN（4 层 dilated causal conv） + 线性分类头。
"""

import torch
import torch.nn as nn


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                              dilation=dilation, padding=0)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = torch.nn.functional.pad(x, (self.padding, 0))
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + residual


class IMUFrameNetV2(nn.Module):
    """
    输入:  (batch, time, 120) — 12 IMU × 10 特征
    输出:  (batch, time, num_classes)
    """
    def __init__(self, input_features=120, num_classes=35, channels=256, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_features, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.tcn = nn.Sequential(
            ResidualTCNBlock(channels, kernel_size=7, dilation=1, dropout=dropout * 0.5),
            ResidualTCNBlock(channels, kernel_size=5, dilation=2, dropout=dropout * 0.5),
            ResidualTCNBlock(channels, kernel_size=5, dilation=4, dropout=dropout * 0.5),
            ResidualTCNBlock(channels, kernel_size=3, dilation=8, dropout=dropout * 0.5),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.input_proj(x)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.classifier(x)
        return x
