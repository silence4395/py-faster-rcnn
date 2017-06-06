
GPU_ID=$1
ITER=$2

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt \
  --net output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_${ITER}.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml