import tensorflow as tf
from tensorflow.python.platform import gfile

sess = tf.Session()
with gfile.FastGFile('./models/la_muse.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='prefix')

sess.run(tf.global_variables_initializer())

for op in sess.graph.get_operations():
    print(op.name)

convert=tf.lite.TFLiteConverter.from_frozen_graph("./models/la_muse.pb",input_arrays=["content"],output_arrays=["Tanh"])
convert.post_training_quantize=True
tflite_model=convert.convert()
open("model.tflite","wb").write(tflite_model)
 
'''
当需要给定输入数据形式时，给出输入格式：
import tensorflow as tf
path="./fullLayer/"
convert=tf.lite.TFLiteConverter.from_frozen_graph(path+"frozen.pb",input_arrays=["images"],output_arrays=["output"],
                                                  input_shapes={"images":[1,540,960,1]})
convert.post_training_quantize=True
tflite_model=convert.convert()
open(path+"quantized_model.tflite","wb").write(tflite_model)
'''
print("finish!")

