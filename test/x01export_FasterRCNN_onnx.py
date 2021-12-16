from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
from torchvision import transforms
import glob
import cv2

if __name__ == '__main__':

    model = fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.5,box_nms_thresh=0.3)
    model.eval()

    transform_func = transforms.ToTensor()

    img_tensor_list = list()
    image_list = list()
    for item in glob.glob("./*.jpg"):
        image_list.append(cv2.resize(cv2.imread(item),dsize=None, fx=0.4, fy=0.4))
        img_tensor_list.append(transform_func(cv2.cvtColor(cv2.resize(cv2.imread(item),dsize=None, fx=0.4, fy=0.4), cv2.COLOR_BGR2RGB)))
        break
    with torch.no_grad():
        result = model(img_tensor_list)
    for i,item in enumerate(result):
        print(item['boxes'].shape)
        print(item['scores'].shape)
        print(item['labels'].shape)
        image = image_list[i].copy()
        for box in item['boxes']:
            box = box.numpy()
            cv2.rectangle(image, tuple(map(int,box[0:2])), tuple(map(int,box[2:4])), (0,255,0))
        cv2.imshow("win", image)
        cv2.waitKey()
        cv2.destroyWindow("win")
    onnx_save_path = "fasterrcnn_backbone_resnet50_fpn_roialign.onnx"
    input_height, input_width = 600, 600
    torch.onnx.export(model, [torch.rand(3,input_height,input_width)], onnx_save_path,
                      verbose=False,
                      do_constant_folding=True,
                      input_names=["input"], output_names=["boxes", "scores", "labels"],
                      # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)

    import onnxsim
    import onnx
    import numpy as np
    model = onnx.load(onnx_save_path)
    # convert model
    
    model_simp, check = onnxsim.simplify(model, check_n=0,input_shapes={'input':[3,input_height,input_width]}, 
                                dynamic_input_shape=False)
    with open(onnx_save_path,'wb') as f:
        onnx.save(model_simp, f)

