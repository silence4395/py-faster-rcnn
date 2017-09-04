#!/bin/bash
#!/bin/sh
# time: 8/31/2017
# author: zhihui.luo@ingenic.com
# 
# Parameter config:
# LRN optional type: POWER AREAS LUT_198 LUT_400
#
#
###############################################################
GPU_ID=0
NET=ZF
DATASET=pascal_voc

bit_width=24
fraction_length=20

LRN_TYPE=LUT_198

function SetLRNType()
{
    awk -v type=$1 -F ' ' '{if (($1 == "op_type") && ($2 == "="))
                               {print " " " " $1 " " $2 " " type ";"}
                            else
                               {print $0;}}' power_layer.cu >| tmp.cu
    mv tmp.cu power_layer.cu
}

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd caffe-fast-rcnn/src/caffe/layers
SetLRNType $LRN_TYPE
cd ../../../
make -j |& tee log

grep "error" log
ERROR=$?
if [ $ERROR -eq 0 ]; then
    exit
fi
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "LRN function type:"
cd src/caffe/layers
grep "op_type =" power_layer.cu
echo "Compiler done!"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

cd ../../../../
./experiments/scripts/lrn_approximate_faster_rcnn_end2end.sh $GPU_ID $NET $DATASET