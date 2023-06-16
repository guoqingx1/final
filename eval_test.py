import argparse
from PIL import Image
from tqdm import tqdm
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab_dense import * 
from modeling.deeplab import *
from torchvision import transforms
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='new_feature_mixed_loss_ASPP')
parser.add_argument('--img', type=str, default='CRACK500_20160222_082414_1281_721.png')
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

checkpoint = torch.load('/content/drive/MyDrive/run/crack/' + run + '/model_best.pth.tar')
model.module.load_state_dict(checkpoint['state_dict'])

image_folder_path = '/content/drive/MyDrive/dataset/Images/'
label_folder_path = '/content/drive/MyDrive/dataset/Masks/'
feature_folder_path = '/content/drive/MyDrive/dataset/Denses/'

model.eval()
print('Model loaded')
transform = transforms.Compose([
    transforms.ToTensor()
])

# image_path = 'CRACK500_20160222_082414_1281_721.png'
image_path = args.img
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

pred = torch.max(output[:3], 1)[1].detach().cpu().numpy()
# print('shape of output: ', pred[0].shape)


# print('pred[0] max value is: ', np.max(pred[0]))
# cv2.imwrite('paper_images/' + rgb_img_path + '.png', pred[0] * 255)
cv2.imwrite('/content/temp.png', pred[0] * 255)
# im.save("paper_images/CFD_001.png")

print('Finished')
