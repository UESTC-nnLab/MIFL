#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
from nets.darknet import vgg
#from nets.MVIT_new import slowfastnet as Model
#from nets.slowfastnet import slowfastnet
from nets.baseline import slowfastnet as Model
if __name__ == "__main__":
    input_shape = [512, 512]
    num_classes = 1
    phi         = 'l'
    
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = Model(num_classes, num_frame=4).to(device)
    summary(m, (3, 4,input_shape[0], 512))
    
    dummy_input     = torch.randn(1, 3, 4,512, 512).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
