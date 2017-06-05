#!/bin/bash
export PYTHONUNBUFFERED="True"

time ./tools/quan_tools/train_net.py --gpu=1 --solver=models/pascal_voc/ZF/faster_rcnn_end2end/solver.prototxt --model=models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt --weights=data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel --trimming_mode=dynamic_fixed_point --model_quantized=models/pascal_voc/ZF/faster_rcnn_end2end/quan_train.prototxt --iters=1 --error_margin=1 --cfg=experiments/cfgs/faster_rcnn_end2end.yml --imdb=voc_2007_trainval