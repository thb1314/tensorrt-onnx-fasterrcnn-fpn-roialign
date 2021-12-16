import torch
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from model import fasterrpn_resnet50_fpn
import glob
from torchvision import transforms
import cv2


if __name__ == '__main__':
    model = fasterrpn_resnet50_fpn(pretrained=True)
    model.eval()

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
        results = model(img_tensor_list, is_show=True)
    result = results[0]
    for i, item in enumerate(result):
        image = image_list[i].copy()
        for score_box in item:
            box = score_box[1:]
            box = box.numpy()
            cv2.rectangle(image, tuple(map(int, box[0:2])), tuple(map(int, box[2:4])), (0, 255, 0))
        cv2.imshow("win", image)
        cv2.waitKey()
        cv2.destroyWindow("win")

    output_names = ["rpn_boxes", *tuple(['feature_'+item for item in results[1].keys()])]

    dynamic_axes = None
    onnx_save_path = 'rpn_backbone_resnet50.onnx'
    torch.onnx.export(model, torch.rand(1, 3, input_height, input_width), onnx_save_path, verbose=False,
                      do_constant_folding=True,
                      input_names=["input"], output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=11)

    import onnxsim
    import onnx
    model = onnx.load(onnx_save_path)
    # convert model
    model_simp, check = onnxsim.simplify(model, check_n=0,input_shapes={'input':[1,3,input_height,input_width]}, 
                                dynamic_input_shape=False)
    with open(onnx_save_path,'wb') as f:
        onnx.save(model_simp, f)

