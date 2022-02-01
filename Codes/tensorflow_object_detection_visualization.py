import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils

def load_image_into_numpy_array(path):

    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(io.BytesIO(image_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16)):
      
    image__np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(image__np_with_annotations,
                                                      boxes,
                                                      classes,
                                                      scores,
                                                      category_index,
                                                      use_normalized_coordinates=False,
                                                      min_score_thresh=0.0)
    
    plt.figure(figsize=figsize)
    plt.imshow(image__np_with_annotations)
    
    
boxes = []
classes = []
scores = []
category_index = {1: {'id': 1, 'name': 'fish'}}
for box in boxes_list:
    x = box['x']
    y = box['y']
    w = box['width']
    h = box['height']
    b_ = [y, x, y+h, x+w]
    boxes.append(b_)
    classes.append(1)
    scores.append(1.0)

boxes = np.array(boxes)
classes = np.array(classes, dtype=np.int32)
scores = np.array(scores, dtype=np.float32)


image = load_image_into_numpy_array(path)
%matplotlib inline
plot_detections(image, boxes, classes, scores, category_index, figsize=(30, 30))
