import onnx_graphsurgeon as gs
import onnx
import numpy as np


def cutOnnx():
    onnx_save_path = "header.onnx"
    graph = gs.import_onnx(onnx.load(onnx_save_path))

    tensors = graph.tensors()
    tensor = tensors["218"]
    graph.inputs = [graph.inputs[-1], tensor.to_variable(dtype=np.float32, shape=('N', 256, 7, 7))]
    graph.inputs[-1].name = "roialigned_feature"

    graph.outputs = [graph.outputs[0], graph.outputs[-1]]

    shape_score = gs.Constant(name="shape_score", values=np.array((-1, 90), dtype=np.int64))
    shape_boxes = gs.Constant(name="shape_boxes", values=np.array((-1, 90, 4), dtype=np.int64))
    shape_boxes_last_node = gs.Constant(name="shape_boxes_last_node", values=np.array((-1, 91, 4), dtype=np.int64))

    # 这里的Reshape_320和Reshape_322是box和score的上一个reshape节点
    for node in graph.nodes:
        if node.name == "Reshape_320":
            node.inputs[-1] = shape_boxes
        elif node.name == "Reshape_322":
            node.inputs[-1] = shape_score
        # the last second reshape node relative to box output
        elif node.name == "Reshape_308":
            node.inputs[-1] = shape_boxes_last_node
    
    # 添加N,90,4 和 N,90,1的结点
    for item in graph.outputs:
        item.shape.insert(1, 90)
        # print(item.shape)
    for graph_output in graph.outputs:
        graph_output.shape[0] = 'N'
    graph.cleanup()
    new_onnx_filepath = 'new_'+onnx_save_path
    onnx.save(gs.export_onnx(graph), new_onnx_filepath)

    import onnxsim
    model = onnx.load(new_onnx_filepath)
    # convert model
    model_simp, check = onnxsim.simplify(model, check_n=0,input_shapes={'roialigned_feature':[1,256, 7, 7],'proposals':[1,4]}, 
                                dynamic_input_shape=True)

    onnx.save(model_simp, new_onnx_filepath)



if __name__ == '__main__':
    cutOnnx()