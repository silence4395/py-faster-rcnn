#!/usr/bin/python2.7
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : quantization.py
## Authors    : zhluo@aries
## Create Time: 2017-05-25:20:59:14
## Description:
##
##
import _init_paths
from fast_rcnn.train import get_training_roidb
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from fast_rcnn.test import im_detect
from utils.timer import Timer
import caffe
from caffe import Net as net
import argparse
import pprint
import math
import numpy as np
import numpy.random as npr
import roi_data_layer.roidb as rdl_roidb
import cv2
import os, sys
import ctypes

from caffe.proto import caffe_pb2
import google.protobuf as pb2
from test import test_net
from train_v1 import train_net

__DEBUG__ON__="NO"

def __DEBUG__(msg):
    if __DEBUG__ON__ == "YES":
	msg="__DEBUG__:"+msg+"\n"
	sys.stdout.write(msg)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Quantization faster_rcnn network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
			default=0, type=int)
    parser.add_argument('--model', dest='prototxt', help='The model definition protocol buffer text file.',
			default=None, type=str)
    parser.add_argument('--weights', dest='caffemodel', help='The trained weights',
			default=None, type=str)
    parser.add_argument('--trimming_mode', dest='trimming_mode',
			help='Available options: dynamic_fixed_point, minifloat or integer_power_of_2_weights.',
			default='dynamic_fixed_point', type=str)
    parser.add_argument('--model_quantized', dest='quan_model', help='The output path of the quantized net',
			default=None, type=str)
    parser.add_argument('--iterations', dest='iter', help='Optional: The number of iterations to run.',
			default=100, type=int)
    parser.add_argument('--error_margin', dest='margin', help='Optional: the allowed accuracy drop in %',
			default=1.0, type=float)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
			action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
			help='set config keys', default=None,
			nargs=argparse.REMAINDER)
    parser.add_argument('--cfg', dest='cfg_file',
			help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
			help='dataset to test',
			default='voc_2007_test', type=str)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
			action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
			help='max number of detections per image',
			default=100, type=int)
    if len(sys.argv) == 1:
	#parser.print_help()
	usage()
	sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
	imdb = get_imdb(imdb_name)
	print 'Loaded dataset `{:s}` for training'.format(imdb.name)
	imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
	print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
	roidb = get_training_roidb(imdb)
	return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
	for r in roidbs[1:]:
	    roidb.extend(r)
	imdb = datasets.imdb.imdb(imdb_names)
    else:
	imdb = get_imdb(imdb_names)
    return imdb, roidb

def get_max_value(args, cfg, iters):
    # Set train config info from faster_rcnn_end2end.yml
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.IMS_PER_BATCH = 1
    cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
    cfg.TRAIN.RPN_BATCHSIZE = 256
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.BG_THRESH_LO = 0.0
    solver_prototxt = 'models/pascal_voc/ZF/faster_rcnn_end2end/solver.prototxt'
    imdb, roidb = combined_roidb(args.imdb_name)
    output_dir = get_output_dir(imdb)

    net = train_net(solver_prototxt, roidb, output_dir,
		    pretrained_model=args.caffemodel, max_iters=iters)
    return net

def forward(iter_num, prototxt, caffemodel, imdb_name, comp_mode, max_per_image, vis, cfg):
    test_score = 0
    for i in range(0, iter_num):
        print ' [Info ] Execute forward get accuracy, total num: ', iter_num, ', current iter: ', i
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(caffemodel))[0]
        imdb = get_imdb(imdb_name)
        imdb.competition_mode(comp_mode)
        if not cfg.TEST.HAS_RPN:
            imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

        test_score = test_score + test_net(net, imdb, max_per_image=max_per_image, vis=vis)
        del net
    test_score = test_score / iter_num
    return test_score

def quantize_mini_float(args, cfg, new_prototxt, test_score_baseline):
    exp_bits_ = 4
    bitwidth = 16
    index = 0
    
    user_type = np.dtype({
	    'names':['bitwidth', 'accuracy'],
	    'formats':['i', 'f']})
    mini_info = np.zeros(5, dtype=user_type)
    
    max_in = caffe.getmaxin()
    max_out = caffe.getmaxout()
    for i in range(0, len(max_in)):
        exp_in = math.ceil(np.log2(np.log2(max_in[i]) - 1) + 1)
        exp_out = math.ceil(np.log2(np.log2(max_out[i]) - 1) + 1)
        exp_bits_ = max(exp_bits_, exp_in, exp_out)
    print ' [Info] Select exponent bit: ', exp_bits_
    
    caffe.minifloat(str(args.prototxt), bitwidth, int(exp_bits_), str(new_prototxt))
    mini_info[index]['bitwidth'] = 16
    mini_info[index]['accuracy'] = forward(args.iter, new_prototxt, args.caffemodel, args.imdb_name, \
                                          args.comp_mode, args.max_per_image, args.vis, cfg)
    index = index + 1
    for i in [8, 4, 2, 1]:
        bitwidth = i
        if (bitwidth - 1 - int(exp_bits_)) > 0:
            caffe.minifloat(str(new_prototxt), bitwidth, int(exp_bits_), str(new_prototxt))
            mini_info[index]['bitwidth'] = bitwidth
            mini_info[index]['accuracy'] = forward(args.iter, new_prototxt, args.caffemodel, args.imdb_name, \
                                                      args.comp_mode, args.max_per_image, args.vis, cfg)
            index = index + 1

    best_bit_width = 32
    for i in range(0, 5):
        if (args.margin/100 >= test_score_baseline - mini_info[i]['accuracy']):
            best_bit_width = mini_info[i]['bitwidth']

    caffe.minifloat(str(args.prototxt), int(best_bit_width), int(exp_bits_), str(args.quan_model))

    print '------------------------------'
    print 'Network accuracy analysis for'
    print 'Convolutional (CONV) and fully'
    print 'connected (FC) layers.'
    print 'Baseline 32bit float: ', test_score_baseline
    print 'Minifloat net:'
    print '16bit: \t', mini_info[0]['accuracy']
    print '8 bit: \t', mini_info[1]['accuracy']
    print '4 bit: \t', mini_info[2]['accuracy']
    print '2 bit: \t', mini_info[3]['accuracy']
    print '1 bit: \t', mini_info[4]['accuracy']
    print 'Select bit width: ', best_bit_width
    print 'Please fine-tune.'

def quantize_dynamic_float(args, cfg, new_prototxt, test_score_baseline):
    # Convolution parameters quantization
    user_type = np.dtype({
	    'names':['bitwidth', 'accuracy'],
	    'formats':['i', 'f']})
    quan_info = np.zeros((3, 5), dtype=user_type) # 16 8 4 2 1

    # Dynamic checkup CONV FC and Layer input and output bit-width
    for qtype in range(0, 3):
	bitwidth = 16
	index = 0
	caffe.dynamicfixfloat(args.prototxt, -1, -1, bitwidth, new_prototxt, qtype)
	quan_info[qtype][index]['bitwidth'] = bitwidth
        quan_info[qtype][index]['accuracy'] = forward(args.iter, new_prototxt, args.caffemodel, args.imdb_name, \
                                                      args.comp_mode, args.max_per_image, args.vis, cfg)
	index = index + 1
	for i in [8, 4, 2, 1]:
	    bitwidth = i
	    caffe.dynamicfixfloat(new_prototxt, -1, -1, bitwidth, new_prototxt, qtype)
            accuracy = forward(args.iter, new_prototxt, args.caffemodel, args.imdb_name, \
                               args.comp_mode, args.max_per_image, args.vis, cfg)
	    if accuracy + args.margin/100 >= test_score_baseline:
		quan_info[qtype][index]['bitwidth'] = bitwidth
		quan_info[qtype][index]['accuracy'] = accuracy
		index = index + 1
	    else:
		break

    # choose best bit-width for different network parts
    best_param_ = [32, 32, 32] # 0: conv weight width, 1: fc weight width, 2: layer input and oputput

    for i in range(0, 3):
	for j in range(0, 5):
	    if (quan_info[i][j]['accuracy'] + args.margin/100 >= test_score_baseline):
		best_param_[i] = quan_info[i][j]['bitwidth']
	    else:
		break

    # Generate quantization prototxt
    # Score dynamic fixed point network.
    # This network combines dynamic fixed point parameters in convolutional and
    # inner product layers, as well as dynamic fixed point activations.
    caffe.dynamicfixfloat(args.prototxt, int(best_param_[0]), int(best_param_[1]), int(best_param_[2]), \
			  args.quan_model, 3)
    accuracy = forward(args.iter, args.quan_model, args.caffemodel, args.imdb_name, \
                       args.comp_mode, args.max_per_image, args.vis, cfg)

    print '------------------------------'
    print 'Network accuracy analysis for'
    print 'Convolutional (CONV) and fully'
    print 'connected (FC) layers.'
    print 'Baseline 32bit float: ', test_score_baseline
    print 'Dynamic fixed point CONV'
    print 'weights: '
    for i in range(0, 5):
	print quan_info[0][i]['bitwidth'], 'bit: \t', quan_info[0][i]['accuracy']
    print 'Dynamic fixed point FC'
    print 'weights: '
    for i in range(0, 5):
	print quan_info[1][i]['bitwidth'], 'bit: \t', quan_info[1][i]['accuracy']
    print 'Dynamic fixed point layer input and output'
    print 'activations: '
    for i in range(0, 5):
	print quan_info[2][i]['bitwidth'], 'bit: \t', quan_info[2][i]['accuracy']
    print 'Dynamic fixed point net:'
    print best_param_[0], ' bit CONV weights'
    print best_param_[1], ' bit FC weights'
    print best_param_[2], ' bit layer activations'
    print ' Accuracy: ', accuracy
    print ' Please fine-tune'

def quantize_power_of_2(args, cfg, test_score_baseline):
    caffe.power_of_two(str(args.prototxt))
    accuracy = forward(args.iter, str(args.prototxt), args.caffemodel, args.imdb_name, \
                           args.comp_mode, args.max_per_image, args.vis, cfg)
    # Write summary of integer-power-of-2-weights analysis to log
    print '------------------------------'
    print 'Network accuracy analysis for'
    print 'Integer-power-of-two weights'
    print 'in Convolutional (CONV) and'
    print 'fully connected (FC) layers.'
    print 'Baseline 32bit float: ', test_score_baseline
    print 'Quantized net:'
    print '4bit: \t', accuracy
    print 'Please fine-tune.'
    
def usage():
    usage_info=\
"""
usage:%s quantization.py [-h] [--gpu GPU_ID] [--model PROTOTXT]
		       [--weights CAFFEMODEL] [--trimming_mode TRIMMING_MODE]
		       [--model_quantized QUAN_MODEL] [--iterations ITER]
		       [--error_margin MARGIN]
"""%(os.path.basename(__file__))
    sys.stdout.write(usage_info)
    sys.exit()

if __name__=="__main__":
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
	cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
	cfg_from_list(args.set_cfgs)
    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
	print('Waiting for {} to exist...'.format(args.caffemodel))
	time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    test_score_baseline = 0
    # Get baseline accuracy
    test_score_baseline = forward(args.iter, args.prototxt, args.caffemodel, args.imdb_name, \
                                  args.comp_mode, args.max_per_image, args.vis, cfg)
    print ' [ Info ] Baseline Score: ', test_score_baseline

    # Get every layer's max value
    net = get_max_value(args, cfg, iters=100)
    net._display_max_value

    new_prototxt = 'models/pascal_voc/ZF/faster_rcnn_end2end/tmp.prototxt'
    if args.trimming_mode == 'minifloat':
	print ' Set minifloat mode.'
	quantize_mini_float(args, cfg, new_prototxt, test_score_baseline)
    elif args.trimming_mode == 'dynamic_fixed_point':
	print ' Set dynamic_fixed_point mode.'
	quantize_dynamic_float(args, cfg, new_prototxt, test_score_baseline)
    elif args.trimming_mode == 'integer_power_of_2_weights':
	print ' Set integer_power_of_2_weights mode.'
        quantize_power_of_2(args, cfg, test_score_baseline)
    else:
	print 'Please set trimming_mode: dynamic_fixed_point, minifloat or integer_power_of_2_weights'
	sys.exit(1)
