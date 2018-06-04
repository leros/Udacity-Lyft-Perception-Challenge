import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time, datetime
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

NUM_CLASSES = 3
LABLES = 13
VEHICLES = 10
ROADS = 7
ROADLINES = 6
NONE = 0
NEW_ROADS = 1
NEW_VEHICLES = 2
CUT_TOP = 128
CUT_BOTTOM = 88

    
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def preprocess_labels(label_image):
    """
    Process lables and map lables to new values
    """
    # Create a new single channel label image to modify
    labels_new = np.copy(label_image[:,:,0])

    # Identify lane marking pixels (label is 6)
    lane_marking_pixels = (label_image[:,:,0] == ROADLINES).nonzero()
    # Set lane marking pixels to road (label is 7)
    labels_new[lane_marking_pixels] = ROADS

    # Identify all vehicle pixels
    vehicle_pixels = (label_image[:,:,0] == VEHICLES).nonzero()
    # Isolate vehicle pixels associated with the hood (y-position > 496)
    hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    labels_new[hood_pixels] = NONE

    # Set all irrelevant pixels to 0, all roads(incl. roadlines) to 1, and all vehicles (excl. hood) to 2
    for label in range(LABLES):     
        if label not in [VEHICLES, ROADS, ROADLINES]:
            irrelevant_pixels = (label_image[:,:,0] == label).nonzero()
            labels_new[irrelevant_pixels] = NONE
    labels_new[labels_new == ROADS] = NEW_ROADS
    labels_new[labels_new == VEHICLES] = NEW_VEHICLES

    # Return the preprocessed label image
    return labels_new

def resize_image(image):
    """
    Resize image from 600*800 to (600-cut_top-cut_bottom)*800
    """
    return image[CUT_TOP:-CUT_BOTTOM,:]

# see https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
def one_hot_encoding(processed_lable_image):
    """
    One hot encode lables
    """
    return (np.arange(NUM_CLASSES) == processed_lable_image[...,None]).astype(int)

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        label_paths = glob(os.path.join(data_folder, 'CameraSeg', '*.png'))
        
        label_paths = dict(zip(image_paths, label_paths))
        
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                images.append(resize_image(image))
                
                gt_image_file = label_paths[image_file]
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                gt_image = preprocess_labels(gt_image)
                gt_image = one_hot_encoding(gt_image)
                gt_images.append(resize_image(gt_image))

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
