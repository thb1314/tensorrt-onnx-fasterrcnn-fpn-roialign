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
    img = cv2.imread('./car.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(600, 600))

    normalize = transforms.Compose([
        transforms.ToTensor(),
    ])
    # ori_image / 255
    img_tensor = normalize(img)
    img_input = img_tensor.numpy().astype(np.float32)

    sess = rt.InferenceSession('fasterrcnn_backbone_resnet50_fpn_roialign.onnx')
    input_name = sess.get_inputs()[0].name
    label_names = [sess.get_outputs()[i].name for i in range(3)]

    pred_onnx = sess.run(label_names, {input_name:img_input})
    pred_onnx = dict(zip(label_names, pred_onnx))

    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box in pred_onnx['boxes']:
        cv2.rectangle(image, tuple(map(int,box[0:2])), tuple(map(int,box[2:4])), (0,255,0))
    cv2.imshow("win", image)
    cv2.waitKey()
    cv2.destroyWindow("win")


