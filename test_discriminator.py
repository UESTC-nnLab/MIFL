import json
import os
import colorsys
#from nets.slowfastnet import slowfastnet as Model
#from nets.MVIT import slowfastnet as Model
from nets.MIFL_base import slowfastnet as Model
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_outputs, non_max_suppression
import torchvision.models as models
#---------------------------------------------------------------------------#
#   map_mode用于指定该文件运行时计算的内容
#   map_mode为0代表整个map计算流程，包括获得预测结果、计算map。
#   map_mode为1代表仅仅获得预测结果。
#   map_mode为2代表仅仅获得计算map。
#---------------------------------------------------------------------------#
map_mode            = 0
#-------------------------------------------------------#
#   指向了验证集标签与图片路径
#-------------------------------------------------------#
#cocoGt_path         = '/home/public/DAUB/annotations/instances_test2017.json'
#cocoGt_path         = '/home/public/ITSDT/annotations/instances_test2017.json'
#cocoGt_path         = '/home/public/IRDST/real/annotations/instances_test2017.json'
#dataset_img_path    = '/home/public/DAUB/'
#dataset_img_path    = '/home/public/ITSDT/'
#dataset_img_path    = '/home/public/IRDST/real/'
cocoGt_path         = '/home/bennyzhu/ILISTO/fewshot/val_all.json'
#-------------------------------------------------------#
#   结果输出的文件夹，默认为map_out
#-------------------------------------------------------#
temp_save_path      = 'map_out/coco_eval'

class MAP_vid(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------/------------------#
        "model_path_1"        : '/home/bennyzhu/ILISTO/logs/metric_cmp/ITSDT_E/best/', 
        "model_path_2"        : '/home/bennyzhu/ILISTO/logs/metric_cmp/DAUB_E/best/', 
        "model_path_3"        : '/home/bennyzhu/ILISTO/logs/metric_cmp/IRDST_E/best/',
        "classes_path"      : 'model_data/classes.txt',
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [512, 512],
        #---------------------------------------------------------------------#
        #   所使用的版本。nano、tiny、s、m、l、x
        #---------------------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        self.net1    = Model(self.num_classes, num_frame=4).loss_s.to('cuda')
        self.net2    = Model(self.num_classes, num_frame=4).loss_s.to('cuda')
        self.net3    = Model(self.num_classes, num_frame=4).loss_s.to('cuda')
        #vgg19 = models.vgg19(pretrained=True).to('cuda')
        #enc_layers = list(vgg19.features.children())
        #self.enc = nn.Sequential(*enc_layers[:12])  # input -> relu1_1
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net1.load_state_dict(torch.load(self.model_path_1 + 'loss.pth', map_location=device))
        self.net2.load_state_dict(torch.load(self.model_path_2 + 'loss.pth', map_location=device))
        self.net3.load_state_dict(torch.load(self.model_path_3 + 'loss.pth', map_location=device))
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.l1_loss = nn.L1Loss()
        #self.net.modify.load_state_dict(torch.load(self.model_path + 'modify.pth', map_location=device))
        self.net1    = self.net1.eval()
        self.net2    = self.net2.eval()
        self.net3    = self.net3.eval()
     #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def calc_mean_std(self, feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def calc_style_loss(self, input, target):
        #assert (input.size() == target.size())
        input_mean, input_std = self.calc_mean_std(input)
        target_mean, target_std = self.calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def resize_img(self,img):
        image_seq = []
        img = cvtColor(img)
        iw, ih = img.size
        w,h = self.input_shape
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        img = img.resize((nw, nh), Image.Resampling.BICUBIC)  # 原图等比列缩放
        new_img = Image.new('RGB', (w,h), (128, 128, 128))  # 预期大小的灰色图
        new_img.paste(img, (dx, dy))  # 缩放图片放在正中
        img = np.transpose(preprocess_input(np.array(new_img, dtype='float32')), (2, 0, 1))
        img = np.expand_dims(img, 0)
        with torch.no_grad():
            img = torch.from_numpy(img)
            if self.cuda:
                img = img.cuda()
        #print(img.shape)
        return img

    def detect_image(self, image_id, images, results, task):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(images[0])[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        img = images[-1]
        
        img = self.resize_img(img)

        images       = [cvtColor(image) for image in images]
        image_data  = [resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image) for image in images]
        image_data = [np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)) for image in image_data]
        # (3, 640, 640) -> (3, 16, 640, 640)
        image_data = np.stack(image_data, axis=1)
        
        image_data  = np.expand_dims(image_data, 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #print(images.shape)
            #---------------------------------------------------------#
            loss1 = self.net1(img)
            loss2 = self.net2(img)
            loss3 = self.net3(img)
            print(loss1,loss2,loss3)
            if loss1 < loss2 and loss1 < loss3:
                min_index = 1  # loss1 最小
            elif loss2 < loss1 and loss2 < loss3:
                min_index = 2  # loss2 最小
            elif loss3 < loss1 and loss3 < loss2:
                min_index = 3  # loss3 最小
            
            if min_index == task:
                return 1
            else:
                return 0
    

def get_history_imgs(line):
    dir_path = line.replace(line.split('/')[-1],'')
    file_type = line.split('.')[-1]
    index = int(line.split('/')[-1][:-4])
    
    return [os.path.join(dir_path,  "%d.%s" % (max(id, 0),file_type)) for id in range(index - 3, index + 1)]


if __name__ == "__main__":
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)
    yolo = MAP_vid(confidence = 0.001, nms_iou = 0.65)
    cocoGt      = COCO(cocoGt_path)
    ids         = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()
    results = 0
    for image_id in tqdm(ids):
        image_path  = os.path.join(cocoGt.loadImgs(image_id)[0]['file_name'])
        if image_path.__contains__("ITSDT"):
            task = 1
        elif image_path.__contains__("DAUB"):
            task = 2
        elif image_path.__contains__("IRDST"):
            task = 3
        images = get_history_imgs(image_path)
        images = [Image.open(item) for item in images]
        # image       = Image.open(image_path)
        results += yolo.detect_image(image_id, images, results, task)
    print(results)

