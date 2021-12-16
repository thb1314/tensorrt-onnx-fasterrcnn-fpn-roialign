
import onnx_graphsurgeon as gs
import onnx
import numpy as np

# def getElementByName(graph, )

def cutOnnx():
    onnx_save_path = "rpn_backbone_resnet50.onnx"
    graph = gs.import_onnx(onnx.load(onnx_save_path))
    print(graph.outputs)
    graph.outputs = graph.outputs[0:-1]
    print(graph.outputs)
    graph.cleanup()
    # remove feature pool
    onnx.save(gs.export_onnx(graph), onnx_save_path)

    


if __name__ == '__main__':
    cutOnnx()