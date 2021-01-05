import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from utils.preprocessing import mean_image_subtraction_numpy
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import cv2


def test(img, gt, pred_out_in, imgIn, sess, threshold=0.5):
    input_shape = 128
    clip = []
    for _x in range(img.shape[0] // input_shape):
        for _y in range(img.shape[1] // input_shape):
            clip.append(img[_x * input_shape:(_x + 1) * input_shape, _y * input_shape:(_y + 1) * input_shape, :])
    clip = np.array(clip)
    pred_numpy = sess.run(pred_out_in, feed_dict={imgIn: clip})
    pred_numpy = np.squeeze(pred_numpy)
    if pred_numpy.shape != (input_shape, input_shape):
        bug = np.concatenate((
            np.concatenate(pred_numpy[0:4], axis=1),
            np.concatenate(pred_numpy[4:8], axis=1),
            np.concatenate(pred_numpy[8:12], axis=1),
            np.concatenate(pred_numpy[12:16], axis=1)),
            axis=0
        )
        pred_numpy = np.array(bug)
    
    if isinstance(threshold, float):
        pred_numpy = pred_numpy > threshold
    elif threshold == 'otasu':
        pred_out2 = pred_numpy.copy()
        pred_out2[:8, :] = 0
        pred_out2[-8:, :] = 0
        pred_out2[:, :8] = 0
        pred_out2[:, -8:] = 0
        V, _ = cv2.threshold((pred_out2 * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, pred_out2 = cv2.threshold((pred_numpy * 255).astype(np.uint8), V, 255, cv2.THRESH_BINARY)
        pred_numpy = pred_out2 > 128
    
    k = np.ones((5, 5))
    if len(gt.shape) == 3:
        real_small = np.mean(gt, axis=-1) > 128
    else:
        real_small = gt
    
    real = real_small
    I = pred_numpy & real
    I_num = np.sum(I)
    U = (np.logical_not(real) & pred_numpy) | real_small
    U_num = np.sum(U)
    IoU = I_num / U_num
    
    real = np.mean(cv2.dilate(gt, k.astype(np.uint8), 1), axis=-1) > 128
    I = pred_numpy & real
    I_num = np.sum(I)
    U = (np.logical_not(real) & pred_numpy) | real_small
    U_num = np.sum(U)
    rIoU = I_num / U_num
    return IoU, rIoU


tf.reset_default_graph()
tf.get_logger().setLevel('ERROR')
IMAGE_SET = Path('/content/tensorflow-deeplab-v3-plus/train')
LABEL_SET = Path('/content/tensorflow-deeplab-v3-plus/train_labels')
CKPT_PATH = './model/model.ckpt-10654'
saver = tf.train.import_meta_graph(CKPT_PATH + '.meta')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver.restore(sess, CKPT_PATH)

_, image_in, _ = tf.get_default_graph().get_operation_by_name('IteratorGetNext').outputs
pred_out = tf.get_default_graph().get_tensor_by_name('softmax_tensor:0')[..., 1]


def aux(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype(np.float32)
    img = mean_image_subtraction_numpy(np.array(img).astype(np.float32))
    return img


def aux2(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Test Set
image_set = sorted([i for i in IMAGE_SET.glob('**/*.*') if i.suffix in ['.jpg', '.png', '.bmp']])
bar = tqdm(total=len(image_set))
image_set = list(map(aux, image_set))
label_set = sorted([i for i in LABEL_SET.glob('**/*.*') if i.suffix in ['.jpg', '.png', '.bmp']])
label_set = list(map(aux2, label_set))

Iou_Set, rIou_Set = [], []
idx = 0
for imgs, labels in zip(image_set, label_set):
    iou, riou = test(imgs, labels, pred_out, image_in, sess, threshold=0.85)
    Iou_Set.append(iou)
    rIou_Set.append(riou)
    bar.update()
bar.close()

print('Mean IoU is:', np.mean(Iou_Set))
print('Mean rIoU is:', np.mean(rIou_Set))