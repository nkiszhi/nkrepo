from models.seqnet.network import *
from models.malconv_gct.network import MalConv
from models.malconv_gct.network import MalConvGCT
from models.regionnet.network import *
from models.model import MobileNet
from thop import profile
from thop import clever_format 
import torch 

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"

# model = SequenceNetFool()
# input = torch.randn(1, 1, 1, 2 ** 18)

# flops, params = profile(model, input)

# flops, params = clever_format([flops, params], "%.10f")

# print('SeqNetFool')
# print('Flops:  ', flops)
# print('Params: ', params)

# model = MalConv().to(device)
# input = torch.randint(0, 1, (1, 2000000, 1))

# flops, params = profile(model, input)

# flops, params = clever_format([flops, params], "%.10f")

# print('Flops:  ', flops)
# print('Params: ', params)


# model = MalConvGCT().to(device)
# input = torch.randint(0, 1, (1, 2 ** 18, 1))

# flops, params = profile(model, input)

# flops, params = clever_format([flops, params], "%.10f")

# print('Flops:  ', flops)
# print('Params: ', params)

# model = MobileNet().to(device)
# input = torch.randn(1, 3, 224, 224)

# flops, params = profile(model, (input,), verbose=False)

# flops, params = clever_format([flops, params], "%.10f")

# print('Flops:  ', flops)
# print('Params: ', params)


# import torch
# from torchvision import models
# # from thop.profile import profile


# print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
# print("---|---|---")




# model = models.mobilenet_v3_small(num_classes=2).cuda()
# dsize = (1, 3, 292, 292)

# inputs = torch.randn(dsize).cuda()
# total_ops, total_params = profile(model, (inputs,), verbose=False)
# print(
#     "%s | %.1f | %.1f" % ("MobileNetV3Small", total_params, total_ops)
# )

# # seqnet2d
# from torchvision import models
# # from thop.profile import profile


# print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
# print("---|---|---")




model = RegionNet()
dsize = (1, 1, 512, 512)
inputs = torch.randn(dsize)
total_ops, total_params = profile(model, (inputs,), verbose=False)
print(
    "%s | %.1f | %.1f" % ("SeqNet2D", total_params, total_ops)
)

# # seqnet2dconv
# from torchvision import models
# # from thop.profile import profile


# print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
# print("---|---|---")




# model = RegionNetConv().cuda()
# dsize = (1, 1, 512, 512)

# inputs = torch.randn(dsize).cuda()
# total_ops, total_params = profile(model, (inputs,), verbose=False)
# print(
#     "%s | %.1f | %.1f" % ("SeqNet2D", total_params, total_ops)
# )

