import torch
import torch.nn as nn
import timm
class BirdClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=234, pretrained=True):
        super(BirdClassifier, self).__init__()
        
        # 直接使用 timm 创建骨干网络
        # in_chans=1: 自动将原本接收 RGB(3) 的第一层改为接收单通道(1)
        # num_classes=0: 移除原有的分类头，方便我们自定义
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            in_chans=1, 
            num_classes=0, 
            global_pool='' # 留空以便后续自定义池化
        )
        
        # 获取骨干网络输出的特征维度
        # EfficientNet-B0 通常是 1280, ResNet34 是 512
        num_features = self.backbone.num_features
        
        # 自定义池化层：结合平均池化和最大池化（竞赛常用技巧）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头：Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 1, 128, 313]
        features = self.backbone(x) # 提取特征图
        pooled = self.global_pool(features) # 全局池化
        logits = self.classifier(pooled) # 输出 234 维向量
        return logits

if __name__ == "__main__":
    # --- 实例化模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdClassifier(model_name='resnet34', num_classes=234).to(device)

    # 测试输入
    dummy_input = torch.randn(4, 1, 128, 313).to(device)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}") # 应该是 [4, 234]