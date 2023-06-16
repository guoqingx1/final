import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab_dense import * 
from modeling.deeplab import *
from utils.metrics import Evaluator
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='new_feature_mixed_loss_ASPP')
args = parser.parse_args()
run = args.run

if 'ASPP' not in run:
    model = DeepLab(num_classes=2,
                    backbone='resnet_feat',
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False)
else:
    model = DeepLab_dense(num_classes=2,
                        backbone='resnet_feat',
                        output_stride=16,
                        sync_bn=None,
                        freeze_bn=False)
model = torch.nn.DataParallel(model, device_ids=[0])
patch_replication_callback(model)
model = model.cuda()

checkpoint = torch.load('/content/drive/MyDrive/run/crack/'+ run + '/model_best.pth.tar')
model.module.load_state_dict(checkpoint['state_dict'])

model.eval()
print('Model loaded')

evaluator = Evaluator(2)
print('Evaluator Loaded')

# image_folder_path = '/home/ubuntu/pytorch-deeplab-xception/CRACK_DETECTION-CVI2021/datasets/rgb/ceramic/'
image_folder_path = '/content/drive/MyDrive/dataset/Images/'
# label_folder_path = '/home/ubuntu/pytorch-deeplab-xception/CRACK_DETECTION-CVI2021/datasets/gt/ceramic/'
label_folder_path = '/content/drive/MyDrive/dataset/Masks/'
feature_folder_path = '/content/drive/MyDrive/dataset/Denses/'

evaluator.reset()
test_images = os.listdir(image_folder_path)
tbar = tqdm(test_images, desc='\r')
test_loss = 0.0

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
])

for i, image_path in enumerate(tbar):
    image = Image.open(image_folder_path + image_path).convert('RGB')
    label = Image.open(label_folder_path + image_path).convert('L')
    feature = Image.open(feature_folder_path + image_path).convert('RGB')

    image = transform(image)
    label = transform(label)
    feature = transform(feature)[1,:,:].unsqueeze(0)


    image, label, feature = image.cuda(), label.cuda(), feature.cuda()

    final_input = torch.cat((image, feature), dim=0).unsqueeze(0)

    with torch.no_grad():
        output = model(final_input)


    pred = output.data.cpu().numpy()
    label = label.cpu().numpy()
    pred = np.argmax(pred, axis=1)

    evaluator.add_batch(label, pred)



mIoU = evaluator.Mean_Intersection_over_Union()
F1 = evaluator.F1_Score()

print('miou: ', mIoU)
print('F1: ', F1)

with open('/content/drive/MyDrive/eval_met/eval_result.txt', 'w') as f:
    import datetime
    now = datetime.datetime.now()
    f.write('Current time: ' + str(now) + '\n')
    f.write('mIoU: ' + str(mIoU) + '\n')
    f.write('F1: ' + str(F1) + '\n')
