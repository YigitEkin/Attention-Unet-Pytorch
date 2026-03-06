import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfAttention(nn.Module):
  def __init__(self, in_dim):
      super(SelfAttention, self).__init__()
      self.chanel_in = in_dim

      self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
      self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
      self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)

      self.gamma = nn.Parameter(torch.zeros(1))

  def forward(self, x):
      m_batchsize, C, width, height = x.size()
      proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
      proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
      energy = torch.bmm(proj_query, proj_key)
      attention = torch.softmax(energy, dim = -1)
      proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

      out = torch.bmm(proj_value, attention.permute(0, 2, 1))
      out = out.view(m_batchsize, C, width, height)

      out = self.gamma*out + x
      return out, attention

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels=None):
      super().__init__()
      if not mid_channels:
          mid_channels = out_channels
      self.double_conv = nn.Sequential(
          nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),                                   nn.BatchNorm2d(mid_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_channels),                                                                                 nn.ReLU(inplace=True)
      )

  def forward(self, x):
      return self.double_conv(x)

class Down(nn.Module):
  def __init__(self, in_channels, out_channels):
      super().__init__()
      self.maxpool_conv = nn.Sequential(
          nn.MaxPool2d(2),
          DoubleConv(in_channels, out_channels)
	  )

  def forward(self, x):
	  return self.maxpool_conv(x)

class Up(nn.Module):
  def __init__(self, in_channels, out_channels, bilinear=True):
      super().__init__()
      if bilinear:
          self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
          self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
      else:
          self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
          self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
      x1 = self.up(x1)
      diffY = x2.size()[2] - x1.size()[2]
      diffX = x2.size()[3] - x1.size()[3]

      x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
      x = torch.cat([x2, x1], dim=1)
      return self.conv(x)


class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
      super(OutConv, self).__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
      return self.conv(x)


class Network(nn.Module):
    def __init__(self, n_channels, n_classes, middle_layers="64,128,256,512,1024", bilinear=False):


        print(middle_layers)
        try: 
            middle_layers = np.array(middle_layers.split(",")).astype(int)
        except:
            raise ValueError("Expected 'middle_layers' to be comma-delimited list of integers like '64,128,256,512,1024'")
        if len(middle_layers) != 5:
            raise ValueError("middle_layers must be a list of length 5")

        super(Network, self).__init__()
        self.n_channels = int(n_channels)
        self.middle_layers = middle_layers
        self.n_classes = int(n_classes)
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, middle_layers[0]))
        self.down1 = (Down(middle_layers[0], middle_layers[1]))
        self.down2 = (Down(middle_layers[1], middle_layers[2]))
        self.down3 = (Down(middle_layers[2], middle_layers[3]))
        self.attention_1 = SelfAttention(middle_layers[0])
        self.attention_2 = SelfAttention(middle_layers[1])
        self.attention_3 = SelfAttention(middle_layers[2])
        self.attention_4 = SelfAttention(middle_layers[3])

        factor = 2 if bilinear else 1
        self.down4 = (Down(middle_layers[3], middle_layers[4] // factor))
        self.up1 = (Up(middle_layers[4], middle_layers[3] // factor, bilinear))
        self.up2 = (Up(middle_layers[3], middle_layers[2] // factor, bilinear))
        self.up3 = (Up(middle_layers[2], middle_layers[1] // factor, bilinear))
        self.up4 = (Up(middle_layers[1], middle_layers[0], bilinear))
        self.outc = (OutConv(middle_layers[0], n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        v1, _ = self.attention_4(x4)
        v2, _ = self.attention_3(x3)
        v3, _ = self.attention_2(x2)
        v4, _ = self.attention_1(x1)
        x = self.up1(x5, v1)
        x = self.up2(x, v2)
        x = self.up3(x, v3)
        x = self.up4(x, v4)
        logits = self.outc(x)
        return logits


def main():

    #from torchsummary import summary
    import optparse

    parser = optparse.OptionParser()
    parser.add_option("-r", "--rows", help="Pixel height of each channel", default=128)
    parser.add_option("-c", "--cols", help="Pixel width of each channel", default=128)
    parser.add_option("-i", "--in_channels", help="Number of input channels", default=6)
    parser.add_option("-o", "--out_channels", help="Number of output channels", default=4)
    parser.add_option(      "--hidden_channels",
        help="Comma-delimited list of hidden layer sizes (e.g. '64,128,256,512,1024')",
        default="64,128,256,512,1024")
    opts, _ = parser.parse_args()


    rows = int(opts.rows)
    cols = int(opts.cols)
    n_channels = int(opts.in_channels)
    n_classes = int(opts.out_channels)
    hidden_channels = opts.hidden_channels

    # Init model 
    model = Network(n_channels, n_classes, hidden_channels)
    print(model)

    # Ensure it works by making a single prediction 
    model.eval() 
    x = torch.ones(1, n_channels, rows, cols)
    y_ = model(x)

    print("")
    print(f"Input batch shape: {x.shape}")
    print(f"Output batch shape: {y_.shape}")


if __name__ == "__main__":
    main()
