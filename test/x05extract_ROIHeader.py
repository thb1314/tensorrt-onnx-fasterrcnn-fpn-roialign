import torch
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from model import fasterrpn_resnet50_fpn, fasterroiheader_resnet50_fpn
import math
import glob
from torchvision import transforms
import cv2
import os


if __name__ == '__main__':
    model = fasterrpn_resnet50_fpn(pretrained=True)
    model_header = fasterroiheader_resnet50_fpn(pretrained=True, transform=model.transform, box_score_thresh=0.5,box_nms_thresh=0.3)
    model.eval()
    model_header.eval()
    img_tensor_list = list()

    transform_func = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_height, input_width = (600 + 31) // 32 * 32, (800 + 31) // 32 * 32
    image_list = list()
    for item in glob.glob("./*.jpg"):
        image_list.append(cv2.resize(cv2.imread(item), dsize=(input_width, input_height)))
        img_tensor_list.append(
            transform_func(cv2.cvtColor(cv2.resize(cv2.imread(item), dsize=(input_width, input_height)), cv2.COLOR_BGR2RGB)))


    with torch.no_grad():
        proposals, features, images, original_image_sizes = model(img_tensor_list)
        if not os.path.exists('buffle.pkl'):
            with open('buffle.pkl', 'wb') as f:
                torch.save({
                    'proposals':proposals,
                    'features':features,
                    'image_sizes':images.image_sizes
                }, f)
        for feature in features.values():
            print('feature.shape',feature.shape)
        proposals = [item[:,1:] for item in proposals]
        if not os.path.exists('roi_result.pkl'):
            roi_result = model_header.roi_heads.box_roi_pool(features, proposals, images.image_sizes)

            with open('roi_result.pkl', 'wb') as f:
                torch.save({
                    'roi_result':roi_result
                }, f)

            dummy_image = torch.rand(1, 3, input_height, input_width)
            batch_size = int(dummy_image.size(0))
            dummy_proposals = [torch.rand((model.rpn.post_nms_top_n(), 4)) for _ in range(batch_size)]
            height, width = int(dummy_image.size(2)), int(dummy_image.size(3))
            dummy_features = {
                key: torch.rand(batch_size, model.backbone.out_channels, math.ceil(height / (2 ** (i + 2))),
                                math.ceil(width / (2 ** (i + 2)))) for i, key in enumerate(features.keys())}

            input_names = [*tuple(['feature_' + key for key in dummy_features.keys()]), 'proposals']

            dynamic_axes = {'proposals': {0: "N"}}
            dynamic_axes.update({'feature_'+key: {0: "B"} for key in dummy_features.keys()})
            dynamic_axes.update({name: {0: "N"} for name in ['outputs']})

            class Wrapper(torch.nn.Module):

                def __init__(self, image_sizes, model):
                    super(Wrapper, self).__init__()
                    self.image_sizes = image_sizes
                    self.model = model
                
                def forward(self, x, boxes):
                    return self.model(x, boxes, self.image_sizes)
            """
            torch.onnx.export(Wrapper(images.image_sizes, model_header.roi_heads.box_roi_pool), (features, dummy_proposals),
                              "roialign.onnx", verbose=True,
                              do_constant_folding=True,
                              input_names=input_names, output_names=["outputs"],
                              dynamic_axes=dynamic_axes,
                              opset_version=11)

            print(roi_result.shape)
            """

        result = model_header(features, proposals, images, original_image_sizes)

    for i, item in enumerate(result):
        image = image_list[i].copy()
        for score_box in item['boxes']:
            box = score_box
            box = box.numpy()
            cv2.rectangle(image, tuple(map(int, box[0:2])), tuple(map(int, box[2:4])), (0, 255, 0))
        cv2.imshow("win", image)
        cv2.waitKey()
        cv2.destroyWindow("win")
    output_names = ["boxes", "labels", "scores"]

    dummy_image = torch.rand(1, 3, input_height, input_width)
    batch_size = int(dummy_image.size(0))
    dummy_proposals = [torch.rand((model.rpn.post_nms_top_n(), 4)) for _ in range(batch_size)]
    height,width = int(dummy_image.size(2)),int(dummy_image.size(3))
    dummy_features = {key:torch.rand(batch_size, model.backbone.out_channels, math.ceil(height / (2 ** (i + 2))), math.ceil(width / (2 ** (i + 2)))) for i,key in enumerate(features.keys())}
    print(dummy_features.keys())
    input_names = [*tuple(['feature_'+key for key in dummy_features.keys()]), 'proposals']
    dynamic_axes = {'proposals': {0: "N"}}
    dynamic_axes.update({name: {0: "N"} for name in output_names})
    onnx_save_path = "header.onnx"
    torch.onnx.export(model_header, (dummy_features, dummy_proposals, dummy_image), onnx_save_path, verbose=True,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=11)

