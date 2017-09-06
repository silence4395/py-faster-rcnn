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
GPU_ID=$1
LRN_TYPE=$2
fixed_point=$3
bit_width=$4
fraction_length=$5

NET=ZF
DATASET=pascal_voc

function SetLRNType()
{
    awk -v type=$1 -F ' ' '{if (($1 == "op_type") && ($2 == "="))
                               {print " " " " $1 " " $2 " " type ";"}
                            else
                               {print $0;}}' power_layer.cu >| tmp.cu
    mv tmp.cu power_layer.cu
}

function SetBitWidth()
{
    awk -v fp=$1 -v bw=$2 -v fl=$3 -F ' ' '{if (($1 == "int") && ($2 == "fixed_point") && ($3 == "="))
                                               {print " " " " $1 " " $2 " " $3 " " fp ";"}
                                            else if (($1 == "int") && ($2 == "bit_width") && ($3 == "="))
                                               {print " " " " $1 " " $2 " " $3 " " bw ";"}
                                            else if (($1 == "int") && ($2 == "fl") && ($3 == "="))
                                               {print " " " " $1 " " $2 " " $3 " " fl ";"}
                                            else
                                               {print $0;} }' power_layer.cu >| tmp.cu
    mv tmp.cu power_layer.cu
}

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
cd caffe-fast-rcnn/src/caffe/layers
SetLRNType $LRN_TYPE
SetBitWidth $fixed_point $bit_width $fraction_length
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
grep "int fixed_point =" power_layer.cu
grep "int bit_width =" power_layer.cu
grep "int fl =" power_layer.cu
echo "Compiler done!"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

cd ../../../../
./experiments/scripts/lrn_approximate_faster_rcnn_end2end.sh $GPU_ID $NET $DATASET |& tee log