from network_parts import Inconv, Down, Up, Outconv, Upself
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoDetectionCNN(nn.Module):
    def __init__(self, n_channels=1, n_classes=16, filter_channel=16, sig=False):
        super().__init__()
        self.inc = Inconv(n_channels, filter_channel)

        self.down = nn.ModuleList([Down(filter_channel, filter_channel*2)])
        self.down.append(Down(filter_channel*4, filter_channel*4))
        self.down.append(Down(filter_channel*4, filter_channel*8))
        self.down.append(Down(filter_channel*8, filter_channel*8))

        self.up1 = Up(filter_channel*16, filter_channel*4)
        self.up2 = Up(filter_channel*8, filter_channel*2)
        self.up3_t = Up(filter_channel*4, filter_channel)
        self.up3_tn = Up(filter_channel*4, filter_channel)
        self.up4_t = Up(filter_channel*2, 32)
        self.up4_tn = Up(filter_channel*2, 32)
        self.outc_t = Outconv(32, n_classes, sig=sig)
        self.outc_tn = Outconv(32, n_classes, sig=sig)

    def forward(self, x):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]
        t_enc, tn_enc, dec = ([0] * 2 for _ in range(3))
        enc = [0] * 4

        t_enc[0] = self.inc(x_inp1)

        tn_enc[0] = self.inc(x_inp2)
        t_enc[1] = self.down[0](t_enc[0])
        tn_enc[1] = self.down[0](tn_enc[0])
        enc[0] = torch.cat([t_enc[1], tn_enc[1]], dim=1)
        for i in range(3):
            enc[i + 1] = self.down[i + 1](enc[i])

        dec = self.up1(enc[-1], enc[-2])
        dec = self.up2(dec, enc[-3])

        t_dec = self.up3_t(dec, t_enc[-1])
        tn_dec = self.up3_tn(dec, tn_enc[-1])

        t_dec = self.up4_t(t_dec, t_enc[-2])
        tn_dec = self.up4_tn(tn_dec, tn_enc[-2])

        pred_t = self.outc_t(t_dec)
        pred_tn = self.outc_tn(tn_dec)
        return pred_t, pred_tn

class CoDetectionCNN_Add_SAM2(nn.Module):
    def __init__(self, n_channels=1, n_classes=16, filter_channel=16, sig=False):
        super().__init__()
        self.inc = Inconv(n_channels, filter_channel)

        self.down = nn.ModuleList([Down(filter_channel, filter_channel*2)])
        self.down.append(Down(filter_channel*4, filter_channel*16))
        self.down.append(Down(filter_channel*16, filter_channel*16))
        self.down.append(Down(filter_channel*16, filter_channel*16))

        self.up1 = Up(filter_channel*32, filter_channel*8)
        self.up2 = Up(filter_channel*24, filter_channel*2)
        self.up3_t = Up(filter_channel*4, filter_channel)
        self.up3_tn = Up(filter_channel*4, filter_channel)
        self.up4_t = Up(filter_channel*2, 32)
        self.up4_tn = Up(filter_channel*2, 32)
        self.outc_t = Outconv(32, n_classes, sig=sig)
        self.outc_tn = Outconv(32, n_classes, sig=sig)

    def forward(self, x, x1_sam=None, x2_sam=None):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]
        t_enc, tn_enc, dec = ([0] * 2 for _ in range(3))
        enc = [0] * 4

        t_enc[0] = self.inc(x_inp1)

        tn_enc[0] = self.inc(x_inp2)
        t_enc[1] = self.down[0](t_enc[0])
        tn_enc[1] = self.down[0](tn_enc[0])
        enc[0] = torch.cat([t_enc[1], tn_enc[1]], dim=1)
        for i in range(3):
            enc[i + 1] = self.down[i + 1](enc[i])

        if x1_sam is not None:
            x_sam = [0] * 3
            x_sam[0] = x1_sam[0] + x2_sam[0]
            x_sam[1] = x1_sam[1] + x2_sam[1]
            x_sam[2] = x1_sam[2] + x2_sam[2]

            enc[-1] = enc[-1] + x_sam[-1]
            enc[-2] = enc[-2] + x_sam[-2]
            enc[-3] = enc[-3] + x_sam[-3]

        dec = self.up1(enc[-1], enc[-2])
        dec = self.up2(dec, enc[-3])

        t_dec = self.up3_t(dec, t_enc[-1])
        tn_dec = self.up3_tn(dec, tn_enc[-1])

        t_dec = self.up4_t(t_dec, t_enc[-2])
        tn_dec = self.up4_tn(tn_dec, tn_enc[-2])

        pred_t = self.outc_t(t_dec)
        pred_tn = self.outc_tn(tn_dec)
        return pred_t, pred_tn

class CoDetectionCNN_extra(nn.Module):
    def __init__(self, n_channels=1, n_classes=16, filter_channel=16, sig=False):
        super().__init__()
        self.inc = Inconv(n_channels, filter_channel)

        self.down = nn.ModuleList([Down(filter_channel+32, filter_channel*2)])
        self.down.append(Down(filter_channel*4, filter_channel*4))
        self.identity = nn.ModuleList([])
        self.identity.append(nn.Conv2d(in_channels=128, out_channels=filter_channel*4, kernel_size=1, stride=1, padding=0))
        self.down.append(Down(filter_channel*4, filter_channel*8))
        self.identity.append(nn.Conv2d(in_channels=256, out_channels=filter_channel*8, kernel_size=1, stride=1, padding=0))
        self.down.append(Down(filter_channel*8, filter_channel*8))
        self.identity.append(nn.Conv2d(in_channels=640, out_channels=filter_channel*8, kernel_size=1, stride=1, padding=0))
        self.up1 = Up(filter_channel*16, filter_channel*4)
        self.up2 = Up(filter_channel*8, filter_channel*2)
        self.up3_t = Up(filter_channel*4, filter_channel)
        self.up3_tn = Up(filter_channel*4, filter_channel)
        self.up4_t = Up(filter_channel*2+32, 32)
        self.up4_tn = Up(filter_channel*2+32, 32)
        self.outc_t = Outconv(32, n_classes, sig=sig)
        self.outc_tn = Outconv(32, n_classes, sig=sig)
        
    def forward(self, x, extra_features):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]
        t_enc, tn_enc, dec = ([0] * 2 for _ in range(3))
        enc = [0] * 4
        extra_feature_t, extra_feature_tn = extra_features[0][:,0], extra_features[0][:,1]
        t_enc[0] = self.inc(x_inp1)
        tn_enc[0] = self.inc(x_inp2)
        # Upsample extra feature to match the dimension of encoder output
        extra_feature_t = F.interpolate(extra_feature_t, size=t_enc[0].shape[-2:], mode='bilinear', align_corners=False)
        extra_feature_tn = F.interpolate(extra_feature_tn, size=tn_enc[0].shape[-2:], mode='bilinear', align_corners=False)
        t_enc[0] = torch.cat([t_enc[0], extra_feature_t], dim=1)
        tn_enc[0] = torch.cat([tn_enc[0], extra_feature_tn], dim=1)
        t_enc[1] = self.down[0](t_enc[0])
        tn_enc[1] = self.down[0](tn_enc[0])
        enc[0] = torch.cat([t_enc[1], tn_enc[1]], dim=1)
        # Apply downsampling and concatenate with extra features
        for i in range(3):
            enc[i + 1] = self.down[i + 1](enc[i])
            # Assuming extra_features is a list of tensors with the same batch dimension
            # and spatial dimensions that match the output of the downsampling
            extra_feature_t, extra_feature_tn = extra_features[i][:,0], extra_features[i][:,1] #
            # Upsample extra feature to match the dimension of encoder output
            extra_feature_t = F.interpolate(extra_feature_t, size=enc[i+1].shape[-2:], mode='bilinear', align_corners=False)
            extra_feature_tn = F.interpolate(extra_feature_tn, size=enc[i+1].shape[-2:], mode='bilinear', align_corners=False)
            enc[i+1] = torch.cat([enc[i+1],extra_feature_t,extra_feature_tn], dim=1)
            enc[i+1] = self.identity[i](enc[i+1])
        
        # Apply upsampling
        dec = self.up1(enc[-1], enc[-2])
        dec = self.up2(dec, enc[-3])

        t_dec = self.up3_t(dec, t_enc[-1])
        tn_dec = self.up3_tn(dec, tn_enc[-1])
        t_dec = self.up4_t(t_dec, t_enc[-2])
        tn_dec = self.up4_tn(tn_dec, tn_enc[-2])

        pred_t = self.outc_t(t_dec)
        pred_tn = self.outc_tn(tn_dec)
        return pred_t, pred_tn

class CoDetectionCNN_extra_SAM2(nn.Module):
    def __init__(self, n_channels=1, n_classes=16, filter_channel=16, sig=False):
        super().__init__()
        self.inc = Inconv(n_channels, filter_channel)

        self.down = nn.ModuleList([Down(filter_channel+256, filter_channel*2)])
        self.down.append(Down(filter_channel*4, filter_channel*4))
        self.identity = nn.ModuleList([])
        self.identity.append(nn.Conv2d(in_channels=filter_channel*4+256*2, out_channels=filter_channel*4, kernel_size=1, stride=1, padding=0))
        self.down.append(Down(filter_channel*4, filter_channel*8))
        self.identity.append(nn.Conv2d(in_channels=filter_channel*8+256*2, out_channels=filter_channel*8, kernel_size=1, stride=1, padding=0))
        self.down.append(Down(filter_channel*8, filter_channel*8))
        self.identity.append(nn.Conv2d(in_channels=filter_channel*8+256*2, out_channels=filter_channel*8, kernel_size=1, stride=1, padding=0))
        self.up1 = Up(filter_channel*16, filter_channel*4)
        self.up2 = Up(filter_channel*8, filter_channel*2)
        self.up3_t = Up(filter_channel*4, filter_channel)
        self.up3_tn = Up(filter_channel*4, filter_channel)
        self.up4_t = Up(filter_channel+272, 32)
        self.up4_tn = Up(filter_channel+272, 32)
        self.outc_t = Outconv(32, n_classes, sig=sig)
        self.outc_tn = Outconv(32, n_classes, sig=sig)
        
    def forward(self, x, extra_features_1, extra_features_2):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]
        t_enc, tn_enc, dec = ([0] * 2 for _ in range(3))
        enc = [0] * 4
        extra_feature_t, extra_feature_tn = extra_features_1[0], extra_features_2[0]
        t_enc[0] = self.inc(x_inp1)
        tn_enc[0] = self.inc(x_inp2)
        # Upsample extra feature to match the dimension of encoder output
        extra_feature_t = F.interpolate(extra_feature_t, size=t_enc[0].shape[-2:], mode='bilinear', align_corners=False)
        extra_feature_tn = F.interpolate(extra_feature_tn, size=tn_enc[0].shape[-2:], mode='bilinear', align_corners=False)
        t_enc[0] = torch.cat([t_enc[0], extra_feature_t], dim=1)
        tn_enc[0] = torch.cat([tn_enc[0], extra_feature_tn], dim=1)
        t_enc[1] = self.down[0](t_enc[0])
        tn_enc[1] = self.down[0](tn_enc[0])
        enc[0] = torch.cat([t_enc[1], tn_enc[1]], dim=1)
        # Apply downsampling and concatenate with extra features
        for i in range(3):
            enc[i + 1] = self.down[i + 1](enc[i])
            # Assuming extra_features is a list of tensors with the same batch dimension
            # and spatial dimensions that match the output of the downsampling
            extra_feature_t, extra_feature_tn = extra_features_1[i], extra_features_2[i] #
            # Upsample extra feature to match the dimension of encoder output
            extra_feature_t = F.interpolate(extra_feature_t, size=enc[i+1].shape[-2:], mode='bilinear', align_corners=False)
            extra_feature_tn = F.interpolate(extra_feature_tn, size=enc[i+1].shape[-2:], mode='bilinear', align_corners=False)
            enc[i+1] = torch.cat([enc[i+1],extra_feature_t,extra_feature_tn], dim=1)
            enc[i+1] = self.identity[i](enc[i+1])
        
        # Apply upsampling
        dec = self.up1(enc[-1], enc[-2])
        dec = self.up2(dec, enc[-3])

        t_dec = self.up3_t(dec, t_enc[-1])
        tn_dec = self.up3_tn(dec, tn_enc[-1])
        t_dec = self.up4_t(t_dec, t_enc[-2])
        tn_dec = self.up4_tn(tn_dec, tn_enc[-2])

        pred_t = self.outc_t(t_dec)
        pred_tn = self.outc_tn(tn_dec)
        return pred_t, pred_tn


if __name__ == "__main__":
    import numpy as np
    # from model.model_para import model_structure
    from ptflops import get_model_complexity_info
    # from model.model_para import model_structure
    x = torch.rand((1, 2, 1024, 1024)).cuda()
    # x = torch.rand((1, 2, 520, 520)).cuda()
    net = CoDetectionCNN(n_channels=1, n_classes=16, filter_channel=16,sig=False).cuda()
    # pred_t, pred_tn = net(x)
    # # model_structure(net)
    # print(pred_t.shape)
    # print(pred_tn.shape)
    macs, params = get_model_complexity_info(net, (2, 1024, 1024), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<20}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<20}'.format('Number of parameters: ', params))


    from torchprofile import profile_macs
    macs = profile_macs(net, x)
    print('model flops (G):', macs / 1.e9, 'input_size:', x.shape)