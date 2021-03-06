#!/usr/bin/env python

# imports and basic setup
import os
from glob import glob
import datetime
from time import sleep
import os.path

from google.protobuf import text_format

import caffe

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

from np_array_utils import *
from dream_utils import *
from obsession_utils import *
from img_utils import *
from file_utils import *
from poster import make_post


# Original emotion recognition model
model_path = 'models/bvlc_googlenet/' # substitute your path here
    
dreamer = Dreamer(
    net_fn=model_path + 'deploy.prototxt',
    param_fn=model_path + 'bvlc_googlenet.caffemodel',
    end_level='inception_5b/output'
)

TIME_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"
    
while True:
    if os.path.isfile('security/data/frames/screenshot.jpg'):
        try:
            now = datetime.datetime.now().strftime(TIME_FORMAT)

            base_image = resizearray(np.float32(PIL.Image.open('security/data/frames/screenshot.jpg')), 1024, 768)
            image = resizearray(base_image, 320, 240)
            
            dream = dreamer.deepdream(image, end='inception_3a/output')
            dream = resizearray(dream, 1024, 768)
            
            fromarray(dream - resizearray(image, 1024, 768) + base_image).save('security/data/frames/screenshot-dream.jpg')
            print('Saving dream to "security/data/frames/screenshot-dream.jpg"')

            make_post('I am bot', 'security/data/frames/screenshot-dream.jpg')
        except KeyboardInterrupt as int:
            raise int
    else:
        print('No imput.')
        sleep(0.1)
