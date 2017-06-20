#!/bin/bash
export PYTHONUNBUFFERED="True"

time ./tools/quan_tools/quantization.py --gpu=1 --model=models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt --weights=data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel --trimming_mode=dynamic_fixed_point --model_quantized=models/pascal_voc/ZF/faster_rcnn_end2end/quan_train.prototxt --iterations=3 --error_margin=1 --cfg=experiments/cfgs/faster_rcnn_end2end.yml --imdb=voc_2007_test |& tee dynamic_fixed_point.log