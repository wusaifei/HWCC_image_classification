import ast
import numpy as np
from PIL import Image
from collections import OrderedDict
from model_service.pytorch_model_service import PTServingBaseService
import torch
from torchvision import transforms
input_size = 380
test_transforms = transforms.Compose([
        transforms.Resize(input_size),
		transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class garbage_classify_service(PTServingBaseService):
    def __init__(self, model_name, model_path):
        print('model_name: ', model_name)
        print('model_path: ', model_path)
        # these three parameters are no need to modify
        self.model_name = model_name
        # self.model_path = model_path
        # self.signature_key = 'predict_images'

        # add the input and output key of your pb model here,
        # these keys are defined when you save a pb file
        self.input_key_1 = 'input_img'
        map_location=torch.device('cpu')
        self.model = torch.load(model_path, map_location=map_location)
        self.model.eval()
        self.label_id_name_dict = \
            {
		"0": "工艺品/仿唐三彩",
                "1": "工艺品/仿宋木叶盏",
                "2": "工艺品/布贴绣",
                "3": "工艺品/景泰蓝",
                "4": "工艺品/木马勺脸谱",
                "5": "工艺品/柳编",
                "6": "工艺品/葡萄花鸟纹银香囊",
                "7": "工艺品/西安剪纸",
                "8": "工艺品/陕历博唐妞系列",
                "9": "景点/关中书院",
                "10": "景点/兵马俑",
                "11": "景点/南五台",
                "12": "景点/大兴善寺",
                "13": "景点/大观楼",
                "14": "景点/大雁塔",
                "15": "景点/小雁塔",
                "16": "景点/未央宫城墙遗址",
                "17": "景点/水陆庵壁塑",
                "18": "景点/汉长安城遗址",
                "19": "景点/西安城墙",
                "20": "景点/钟楼",
                "21": "景点/长安华严寺",
                "22": "景点/阿房宫遗址",
                "23": "民俗/唢呐",
                "24": "民俗/皮影",
                "25": "特产/临潼火晶柿子",
                "26": "特产/山茱萸",
                "27": "特产/玉器",
                "28": "特产/阎良甜瓜",
                "29": "特产/陕北红小豆",
                "30": "特产/高陵冬枣",
                "31": "美食/八宝玫瑰镜糕",
                "32": "美食/凉皮",
                "33": "美食/凉鱼",
                "34": "美食/德懋恭水晶饼",
                "35": "美食/搅团",
                "36": "美食/枸杞炖银耳",
                "37": "美食/柿子饼",
                "38": "美食/浆水面",
                "39": "美食/灌汤包",
                "40": "美食/烧肘子",
                "41": "美食/石子饼",
                "42": "美食/神仙粉",
                "43": "美食/粉汤羊血",
                "44": "美食/羊肉泡馍",
                "45": "美食/肉夹馍",
                "46": "美食/荞面饸饹",
                "47": "美食/菠菜面",
                "48": "美食/蜂蜜凉粽子",
                "49": "美食/蜜饯张口酥饺",
                "50": "美食/西安油茶",
                "51": "美食/贵妃鸡翅",
                "52": "美食/醪糟",
                "53": "美食/金线油塔"
            }

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = test_transforms(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data[self.input_key_1]
        img = img[np.newaxis, :, :, :]  # the input tensor shape of resnet is [?, 224, 224, 3]
        # pred_score = self.sess.run([self.output_score], feed_dict={self.input_images: img})
        
        pred_score = self.model(img)
        pred_score = pred_score.detach().numpy()
        if pred_score is not None:
            pred_label = np.argmax(pred_score, axis=1)[0]
            result = {'result': self.label_id_name_dict[str(pred_label)]}
        else:
            result = {'result': 'predict score is None'}
        return result

    def _postprocess(self, data):
        return data
