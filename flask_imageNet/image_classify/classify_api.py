import re

import numpy as np
import tensorflow as tf


class NodeLookup(object):
    """
    convert tag to human readable text
    """
    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            #label_lookup_path = 'models/imagenet/imagenet_2012_challenge_label_map_proto.pbtxt'
            label_lookup_path = 'models/coco/mscoco_label_map.pbtxt'
        if not uid_lookup_path:
            uid_lookup_path = 'models/imagenet/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string
        """
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  id:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  display_name:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]
        print('john123 node_id_to_uid: %s' % node_id_to_uid)
        print('john type %s' % type(node_id_to_uid))
        print('john test dict %s' % node_id_to_uid[2])
        return node_id_to_uid

        """
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name
            # node_id_to_name ={id1:name1, id2:name2, ...}
        return node_id_to_name
        """

    def id_to_string(self, node_id):
        print('john node_id type %s' % type(node_id))
        print('john node_id %s' % node_id)
        print('john node_lookup %s' % self.node_lookup)

        if int(node_id) not in self.node_lookup:
            print('123456')
            print('node_lookup %s' % self.node_lookup)
            return ''
        return self.node_lookup[node_id]


with tf.gfile.FastGFile('models/coco/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

node_lookup = NodeLookup()

sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')


def classify(image_data):
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-10:][::-1]
    print('classify result top_k: %s' % top_k)
    print('classify predictions len: %s' % len(predictions))

    results = []
    for node_id in top_k:
        
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        print('john human_string %s' % human_string)
        # print('%s (score = %.5f)' % (human_string, score))
        results.append({'label': human_string, 'score': '{:.2}'.format(score)})
    print('classify return results: %s' % results)
    return results


if __name__ == '__main__':
    import os

    images = os.listdir('images')
    images_url = ['images/' + name for name in images]
    for image in images_url:
        data = tf.gfile.FastGFile(image, 'rb').read()
        print(classify(data))
