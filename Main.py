import argparse
arg = argparse.ArgumentParser()
arg.add_argument('-d','--dir',required=True, help = "Dataset Directory")
arg.add_argument('-o','--output',required=True, help = "Model output name")
arg.add_argument('-t','--text',required=True, help = "Text or Image (Boolean) ")
args = vars(arg.parse_args())

from CNN_Keras.CNNModel import CNNModel

model = CNNModel()
model.run(args['dir'],args['output'],args['text'])
