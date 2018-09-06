
Parameters = {
    "lr":0.001,
    "epoch" : 20,
    "batch_size" : 110,
    "optimizer" : "adam",
    "loss":"categorical_crossentropy",
    "image_dim":36
}

CoreLayers = {
    "nodes":200,
    "output_node":58,
    "dropout": 0.3,
    "activation":"relu",
    "activation_output": "softmax"

}

ConvLayers ={
    "padding":"same",
    "activation":"relu",
    "filter":36,
    "dropout":0.25,
    "kernel_size":(3,3),
    "pool_size":(2,2)
}