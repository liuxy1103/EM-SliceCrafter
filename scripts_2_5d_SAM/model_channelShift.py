import torch.nn as nn
import torch


class StepByStepUpscaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(128+128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))  # [32, 128, 16, 16]
        x = self.relu(self.conv2(x))  # [32, 64, 32, 32]
        x = self.relu(self.conv3(x))  # [32, 32, 64, 64]
        x = self.conv4(x)             # [32, 16, 128, 128]
        return x


class StepByStepUpscaler4(nn.Module):
    def __init__(self):
        super().__init__()
        # 增大反卷积层的输出通道数
        self.conv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        # 增大1x1卷积层的输入和输出通道数
        self.conv_fuse1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_fuse2 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_fuse3 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_fuse4 = nn.Conv2d(64, 32, kernel_size=1)

        # 添加一个卷积层用于调整到16倍上采样后的尺寸
        self.adjust_conv = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, x1, x2):
        # 第一层
        x1_1 = self.relu(self.conv1(x1))
        x2_1 = self.relu(self.conv1(x2))
        x_fused1 = torch.cat([x1_1, x2_1], dim=1)
        x_fused1 = self.conv_fuse1(x_fused1)

        # 第二层
        x1_2 = self.relu(self.conv2(torch.cat([x_fused1, x1_1], dim=1)))
        x2_2 = self.relu(self.conv2(torch.cat([x_fused1, x2_1], dim=1)))
        x_fused2 = torch.cat([x1_2, x2_2], dim=1)
        x_fused2 = self.conv_fuse2(x_fused2)

        # 第三层
        x1_3 = self.relu(self.conv3(torch.cat([x_fused2, x1_2], dim=1)))
        x2_3 = self.relu(self.conv3(torch.cat([x_fused2, x2_2], dim=1)))
        x_fused3 = torch.cat([x1_3, x2_3], dim=1)
        x_fused3 = self.conv_fuse3(x_fused3)

        # 第四层
        x1_4 = self.relu(self.conv4(torch.cat([x_fused3, x1_3], dim=1)))
        x2_4 = self.relu(self.conv4(torch.cat([x_fused3, x2_3], dim=1)))
        x_fused4 = torch.cat([x1_4, x2_4], dim=1)
        x_fused4 = self.conv_fuse4(x_fused4)

        # 第五层
        x1 = self.conv5(torch.cat([x_fused4, x1_4], dim=1))
        x2 = self.conv5(torch.cat([x_fused4, x2_4], dim=1))

        # 使用调整卷积层
        x1 = self.adjust_conv(x1)
        x2 = self.adjust_conv(x2)

        return x1, x2


class StepByStepUpscaler2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(128+128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv_fuse1 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x1, x2):
        # 对x1进行反卷积和特征处理
        x1 = self.relu(self.conv1(x1))
        x2 = self.relu(self.conv1(x2))
        # 融合x1和x2在第一层的特征，这里简单相加，可按需修改融合方式
        x_fused = torch.cat([x1, x2], dim=1)
        x_fused = self.conv_fuse1(x_fused) #128


        x1 = self.relu(self.conv2(torch.cat([x_fused, x1], dim=1)))
        x1 = self.relu(self.conv3(x1))
        x1 = self.conv4(x1)        
        x2 = self.relu(self.conv2(torch.cat([x_fused, x2], dim=1)))
        x2 = self.relu(self.conv3(x2))
        x2 = self.conv4(x2)        

        return x1, x2


class StepByStepUpscaler3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.ConvTranspose2d(64+64, 32, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(32+32, 16, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv_fuse1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_fuse2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_fuse3 = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x1, x2):
        # 第一层
        x1_1 = self.relu(self.conv1(x1))
        x2_1 = self.relu(self.conv1(x2))
        x_fused1 = torch.cat([x1_1, x2_1], dim=1)
        x_fused1 = self.conv_fuse1(x_fused1)

        # 第二层
        x1_2 = self.relu(self.conv2(torch.cat([x_fused1, x1_1], dim=1)))
        x2_2 = self.relu(self.conv2(torch.cat([x_fused1, x2_1], dim=1)))
        x_fused2 = torch.cat([x1_2, x2_2], dim=1)
        x_fused2 = self.conv_fuse2(x_fused2)

        # 第三层
        x1_3 = self.relu(self.conv3(torch.cat([x_fused2, x1_2], dim=1)))
        x2_3 = self.relu(self.conv3(torch.cat([x_fused2, x2_2], dim=1)))
        x_fused3 = torch.cat([x1_3, x2_3], dim=1)
        x_fused3 = self.conv_fuse3(x_fused3)

        # 第四层
        x1 = self.conv4(torch.cat([x_fused3, x1_3], dim=1))
        x2 = self.conv4(torch.cat([x_fused3, x2_3], dim=1))

        return x1, x2

if __name__ == "__main__":
    import numpy as np
    input = np.random.random((32,256,8,8)).astype(np.float32)
    x = torch.tensor(input).cuda()
    print("x.shape: ", x.shape)

    # model = UNet_PNI_embedding(filters=[28, 36, 48, 64, 80], upsample_mode='bilinear', merge_mode='add')
    model = StepByStepUpscaler().to('cuda')
    out = model(x)
    print("out.shape: ", out.shape)

    # macs, params = get_model_complexity_info(model, (1, 18,160,160), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
