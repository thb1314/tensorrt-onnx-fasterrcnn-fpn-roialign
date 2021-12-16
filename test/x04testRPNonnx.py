import onnxruntime as rt
import numpy as  np
import torch
import torchvision
import cv2
from torchvision import transforms

def get_classes(filepath):
    with open(filepath, 'r', encoding='gbk') as f:
        return [item.strip() for item in f.readlines()]


if __name__ == '__main__':
    onnx_save_path = "rpn_backbone_resnet50.onnx"
    img = cv2.imread('./car.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(800, 608))

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = normalize(img).unsqueeze(dim=0)
    img_input = img_tensor.numpy().astype(np.float32)

    sess = rt.InferenceSession(onnx_save_path)
    input_name = sess.get_inputs()[0].name
    label_names = [sess.get_outputs()[i].name for i in range(1)]

    pred_onnx = sess.run(label_names, {input_name:img_input})
    # output without nms
    pred_onnx = dict(zip(label_names, pred_onnx))

    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box in pred_onnx['rpn_boxes'][0]:
        box = box[1:]
        cv2.rectangle(image, tuple(map(int,box[0:2])), tuple(map(int,box[2:4])), (0,255,0))
    cv2.imshow("win", image)
    cv2.waitKey()
    cv2.destroyWindow("win")


