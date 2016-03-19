import os

import numpy as np
import scipy.ndimage as nd
from google.protobuf import text_format

import caffe

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

from np_array_utils import *
from img_utils import *


# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']


def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


def objective_L2(dst):
    dst.diff[:] = dst.data


def make_step(net, step_size=1.5, end=None,
              jitter=32, clip=True, objective=objective_L2):
        '''Basic gradient ascent step.'''
        # default end layer is a global variable
        if end is None:
            end = DREAM_OPTIONS['end_level']

        src = net.blobs['data'] # input image is stored in Net's 'data' blob
        dst = net.blobs[end]

        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
                
        net.forward(end=end)
        objective(dst)  # specify the optimization objective
        net.backward(start=end)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += step_size/np.abs(g).mean() * g

        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
                
        if clip:
            bias = net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255-bias)


class Dreamer:
    def __init__(self, net_fn, param_fn, end_level, channel_swap=(2,1,0)):
        self.net_fn = net_fn
        self.param_fn = param_fn
        self.end_level = end_level
        self.channel_swap = channel_swap

        # Patching model to be able to compute gradients.
        # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
        self.model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(self.net_fn).read(), self.model)
        self.model.force_backward = True
        open('tmp.prototxt', 'w').write(str(self.model))

        self.net = caffe.Classifier('tmp.prototxt', self.param_fn,
                               mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                               channel_swap = self.channel_swap) # the reference model has channels in BGR order instead of RGB

    def deepdream(self, base_img, iter_n=10, octave_n=4, octave_scale=1.4, resize_out=None,
                  end=None, clip=True, show_diff=False, save_as=None, mask=None, show_results=True,
                  **step_params):
        # default end layer is a class member
        if end is None:
            end = self.end

        # remember image dimensions
        o_h, o_w, _ = base_img.shape

        # calculate sie for intermediate output
        r_w, r_h = o_w, o_h
        if resize_out is not None:
            r_w, r_h = resize_out
        
        # prepare base images for all octaves
        octaves = [preprocess(self.net, base_img)]
        for i in xrange(octave_n-1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
        
        src = self.net.blobs['data']
        detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
        frame_i = 1
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

            src.reshape(1,3,h,w) # resize the network's input image size
            src.data[0] = octave_base+detail
            for i in xrange(iter_n):
                make_step(self.net, end=end, clip=clip, **step_params)
                
                # visualization
                vis = deprocess(self.net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                # resize for better preview
                vis = resizearray(vis, r_w, r_h)
                # apply mask if required
                if mask:
                    vis = apply_mask_to_img(vis, mask)
                # save frame to file:
                if save_as is not None:
                    fromarray(vis).save('%s-%04d-%02d.jpg' % (save_as, frame_i, i))
                # show onli difference if required
                if show_diff:
                    vis = vis - base_img
                    # save frame diff to file:
                    if save_as is not None:
                        fromarray(vis).save('%s-%04d-diff.jpg' % (save_as, frame_i))
                # show result of the stage
                if show_results:
                    showarray(vis)
                    
                # print frame statistics and clear output if necessary
                print(octave, i, end, vis.shape)
                if show_results:
                    clear_output(wait=True)
                
            # extract details produced on the current octave
            detail = src.data[0]-octave_base
            # increment frame counter
            frame_i += 1
            
        # returning the resulting image
        return deprocess(self.net, src.data[0])


    def long_dream(self, base_img, stages=[],
                   resize_in=None, resize_out=None,
                   show_diff=False, save_as=None, mask=None,
                   show_results=True, skip_stages=0, **step_params):
        if save_as is not None:
            fromarray(base_img).save('%s-00-base.jpg'%save_as)
        
        img = base_img
        if resize_in is not None:
            r_w, r_h = resize_in
            img=resizearray(img, r_w, r_h)
            
        for s in xrange(len(stages)):
            stage = stages[s]
            # append stage name
            if save_as is not None:
                save_as += '-%02d-%s' % (s+1, stage)
            # define name for iteration
            save_stage_as = save_as
            # skip stage saving if required
            if s < skip_stages: 
                save_stage_as = None
            # resize at last stage if required
            if s == len(stages) -1 and resize_out is not None:
                r_w, r_h = resize_out
                img=resizearray(img, r_w, r_h)

            img = self.deepdream(img, end=stage, show_diff=show_diff, save_as=save_stage_as, mask=mask,
                                 resize_out=resize_out, show_results=show_results, **step_params)
            
        # apply mask if required
        if mask:
            img = apply_mask_to_img(img, mask)

        return img
