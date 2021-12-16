import onnxruntime as rt
import numpy as  np
import torch
import torchvision
import cv2
from torchvision import transforms


if __name__ == '__main__':

    sess = rt.InferenceSession('new_header.onnx')
    input_names = [item.name for item in sess.get_inputs()]
    output_names = [item.name for item in sess.get_outputs()]

    # proposal = np.array([1,1,10,10], dtype=np.float32).reshape(-1, 4)
    b = 10
    input_dict = dict(
        proposals = np.random.randn(b, 4).astype(dtype=np.float32),
        roialigned_feature = np.random.randn(b, 256, 7, 7).astype(dtype=np.float32)
    )
    pred_onnx = sess.run(output_names, input_dict)
    pred_onnx = dict(zip(output_names, pred_onnx))
    print(pred_onnx['boxes'].shape)
    # print(pred_onnx['boxes'])
    print(pred_onnx['scores'].shape)
    # print(pred_onnx['scores'])

