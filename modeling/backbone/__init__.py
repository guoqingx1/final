from modeling.backbone import resnet, resnet_feat, xception, xception_feat

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    if backbone == 'resnet_feat':
        return resnet_feat.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'xception_feat':
        return xception_feat.AlignedXception(output_stride, BatchNorm)
    else:
        raise NotImplementedError
