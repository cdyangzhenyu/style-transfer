#! /usr/bin/env python3
"""Run from root directory of repo https://github.com/lengstrom/fast-style-transfer to
create a .pb for use with OpenVINO.
"""
import sys
sys.path.insert(0, 'src')
import transform
import argparse
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


def protobuf_from_checkpoint(ckpt_file, image_shape, batch_size, output_name):

    #if not os.path.isfile(ckpt_file):
    #    raise ValueError(f'File "{ckpt_file}" does not exist or is not a file.')

    # create the tf Session
    sess            = tf.Session()
    # Compute the shape of the input placeholder. This shape is what the serialized model can
    # process. For other input shapes you will have to resize the images or make a new model export
    batch_shape     = [batch_size] + image_shape
    img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
    # create the network to the variables are in the global scope
    preds           = transform.Transform().net(img_placeholder)  # noqa
    saver           = tf.train.Saver()
    # load our checkpoint into the variables
    saver.restore(sess, ckpt_file)

    # get the tf graph and retrieve operation names
    graph    = tf.get_default_graph()
    op_names = [op.name for op in graph.get_operations()]
    # convert the protobuf GraphDef to a GraphDef that has no variables but just constants with the
    # current values.
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph.as_graph_def(), op_names)

    # dump GraphDef to file
    graph_io.write_graph(output_graph_def, './', output_name, as_text=False)
    sess.close()


def main():
    # parse required aguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='The checkpoint file to convert.', type=str,
                        required=True,
                        metavar='checkpoint.ckpt')
    parser.add_argument('-s', '--image-shape',
                        help='Shape of the image the network processes (H, W, C)',
                        nargs=3, metavar='size', required=True)
    parser.add_argument('-b', '--batch-size', help='Batch size the network processes', type=int,
                        default=1)
    parser.add_argument('-o', '--output-name', help='Name of the output file. '
                                                    'The name is relative to the current directory.',
                        type=str,
                        default='model.pb')
    args = parser.parse_args()
    protobuf_from_checkpoint(args.file, args.image_shape, args.batch_size, args.output_name)
    print(f'./{args.output_name} written.')


if __name__ == '__main__':
    main()
