from network_parts import Inconv, Down, Up, Outconv, Upself
import torch
import torch.nn as nn


class CoDetectionCNN(nn.Module):
    def __init__(self, n_channels=1, n_classes=3, filter_channel=64, sig=True):
        super().__init__()
        self.inc = Inconv(n_channels, filter_channel)

        self.down = nn.ModuleList([Down(filter_channel, filter_channel*2)])
        self.down.append(Down(filter_channel*4, filter_channel*4))
        self.down.append(Down(filter_channel*4, filter_channel*8))
        self.down.append(Down(filter_channel*8, filter_channel*8))

        self.up1 = Up(filter_channel*16, filter_channel*4)
        self.up2 = Up(filter_channel*8, filter_channel*2)
        self.up3_tn = Up(filter_channel*4, filter_channel)
        self.up4_tn = Up(filter_channel*2, 32)
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

        tn_dec = self.up3_tn(dec, tn_enc[-1])

        tn_dec = self.up4_tn(tn_dec, tn_enc[-2])

        pred_tn = self.outc_tn(tn_dec)
        return pred_tn

if __name__ == "__main__":
    from model.model_para import model_structure
    x = torch.rand((16, 2, 256, 256)).cuda()
    net = CoDetectionCNN(n_channels=1, n_classes=3, filter_channel=16).cuda()
    pred_tn = net(x)
    model_structure(net)
    print(pred_tn.shape)
