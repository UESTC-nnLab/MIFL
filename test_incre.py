import json
import os
import colorsys
#from nets.slowfastnet import slowfastnet as Model
#from nets.MVIT import slowfastnet as Model
from nets.MIFL_incre import slowfastnet as Model_incre
from nets.MIFL_base import slowfastnet as Model_base
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
cocoGt_path         = '/home/bennyzhu/ILISTO/fewshot/val_all.json'
#cocoGt_path         = '/home/public/DAUB/annotations/instances_test2017.json'
#cocoGt_path         = '/home/public/ITSDT/annotations/instances_test2017.json'
#cocoGt_path         = '/home/public/IRDST/real/annotations/instances_test2017.json'
#dataset_img_path    = '/home/public/DAUB/'
#dataset_img_path    = '/home/public/ITSDT/'
#dataset_img_path    = '/home/public/IRDST/real/'
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
        #--------------------------------------------------------------------------#
        "model_path_1"        : '/home/bennyzhu/ILISTO/logs/ITSDT/50_prompt/best/', 
        "model_path_2"        : '/home/bennyzhu/ILISTO/logs/DAUB/50_th10/best/', 
        "model_path_3"        : '/home/bennyzhu/ILISTO/logs/IRDST/loss_2024_11_07_19_54_47/best/',

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
    def load_pth_base(self,model,model_path):
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.backbone.load_state_dict(torch.load(model_path + 'backbone.pth', map_location=device))
        model.neck.load_state_dict(torch.load(model_path + 'neck.pth', map_location=device))
        model.head.load_state_dict(torch.load(model_path + 'head.pth', map_location=device))
        model.loss_s.load_state_dict(torch.load(model_path + 'loss_s.pth', map_location=device))
        model.eval()

    def load_pth_incre(self,model,model_path):
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.backbone.load_state_dict(torch.load(model_path + 'backbone.pth', map_location=device))
        model.neck.load_state_dict(torch.load(model_path + 'neck.pth', map_location=device))
        model.head.load_state_dict(torch.load(model_path + 'head.pth', map_location=device))
        model.loss_s.load_state_dict(torch.load(model_path + 'loss_s.pth', map_location=device))
        model.loss_p.load_state_dict(torch.load(model_path + 'loss_p.pth', map_location=device))
        model.eval()

    def resize_img(self,image):
        image_seq = []
        img = cvtColor(image)
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
    def generate(self, onnx=False):
        self.net1    = Model_incre(self.num_classes, num_frame=4).to('cuda')
        self.net2    = Model_incre(self.num_classes, num_frame=4).to('cuda')
        self.net3    = Model_base(self.num_classes, num_frame=4).to('cuda')
        self.load_pth_incre(self.net1, self.model_path_1)
        self.load_pth_incre(self.net2, self.model_path_2)
        self.load_pth_base(self.net3, self.model_path_3)

        #print('{} model, and classes loaded.'.format(self.model_path))
                
     #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_id, images, results):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        img = self.resize_img(images[-1])
        image_shape = np.array(np.shape(images[0])[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        images       = [cvtColor(image) for image in images]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = [resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image) for image in images]
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
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
            #---------------------------------------------------------#
            loss1 = self.net1.loss_s(img)
            loss2 = self.net2.loss_s(img)
            loss3 = self.net3.loss_s(img)
            losses = [loss1,loss2,loss3]
            models = [self.net1,self.net2,self.net3]
            min_loss_index = losses.index(min(losses))
            self.net = models[min_loss_index].to('cuda')
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
            outputs,_,_ = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            outputs = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if outputs[0] is None: 
                return results

            top_label   = np.array(outputs[0][:, 6], dtype = 'int32')
            top_conf    = outputs[0][:, 4] * outputs[0][:, 5]
            top_boxes   = outputs[0][:, :4]

        for i, c in enumerate(top_label):
            result                      = {}
            top, left, bottom, right    = top_boxes[i]

            result["image_id"]      = int(image_id)
            result["category_id"]   = clsid2catid[c]
            result["bbox"]          = [float(left),float(top),float(right-left),float(bottom-top)]
            result["score"]         = float(top_conf[i])
            results.append(result)
        return results

def get_history_imgs(line):
    dir_path = line.replace(line.split('/')[-1],'')
    file_type = line.split('.')[-1]
    index = int(line.split('/')[-1][:-4])
    
    return [os.path.join(dir_path,  "%d.%s" % (max(id, 0),file_type)) for id in range(index - 3, index + 1)]


if __name__ == "__main__":
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    cocoGt      = COCO(cocoGt_path)
    ids         = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()

    if map_mode == 0 or map_mode == 1:
        yolo = MAP_vid(confidence = 0.001, nms_iou = 0.65)

        with open(os.path.join(temp_save_path, 'eval_results.json'),"w") as f:
            results = []
            for image_id in tqdm(ids):
                image_path  =  image_path  = os.path.join( cocoGt.loadImgs(image_id)[0]['file_name'])
                images = get_history_imgs(image_path)
                images = [Image.open(item) for item in images]
                # image       = Image.open(image_path)
                results     = yolo.detect_image(image_id, images, results)
            json.dump(results, f)

    if map_mode == 0 or map_mode == 2:
        cocoDt      = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))
        cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox') 
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        """
        T:iouThrs [0.5:0.05:0.95] T=10 IoU thresholds for evaluation
        R:recThrs [0:0.01:100] R=101 recall thresholds for evaluation
        K: category ids 
        A: [all, small, meduim, large] A=4 
        M: maxDets [1, 10, 100] M=3 max detections per image
        """
        precisions = cocoEval.eval['precision']
        precision_50 = precisions[0,:,0,0,-1]  # 第三为类别 (T,R,K,A,M)
        recalls = cocoEval.eval['recall']
        recall_50 = recalls[0,0,0,-1] # 第二为类别 (T,K,A,M)

        
        print("Precision: %.4f, Recall: %.4f, F1: %.4f" %(np.mean(precision_50[:int(recall_50*100)]), recall_50, 2*recall_50*np.mean(precision_50[:int(recall_50*100)])/( recall_50+np.mean(precision_50[:int(recall_50*100)]))))
        print("Get map done.")
        with open("Ours.txt", 'w') as f:
             for pred in precision_50:
                 f.writelines(str(pred)+'\t')
        # 画图
        #import matplotlib.pyplot as plt
        #plt.figure(1) 
        #plt.title('PR Curve')# give plot a title
        #plt.xlabel('Recall')# make axis labels
        #plt.ylabel('Precision')
        
        #x_axis = plt.xlim(0,100)
        #y_axis = plt.ylim(0,1.05)
        #plt.figure(1)
        #plt.plot(precision_50)
        #plt.show()
        #plt.savefig('p-r.png')