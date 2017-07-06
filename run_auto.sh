#!/bin/bash
#
# author: zhluo@ingenic.com
# time: 22/6/2017
########################################################

max_point=0.7
svae_point=0.55
end_point=0.594
max_accuracy=0
iter=0
epoch=20

./experiments/scripts/faster_rcnn_end2end_separable_conv.sh 0 ZF pascal_voc |& tee faster_rcnn.log

accuracy=`cat faster_rcnn.log | grep -oP 'Mean AP = \K\S+'`
echo 'current accuracy: '$accuracy

while [[ $accuracy < $max_point ]]
do
   ./experiments/scripts/faster_rcnn_end2end_separable_conv.sh 0 ZF pascal_voc |& tee faster_rcnn.log
    accuracy=`cat faster_rcnn.log | grep -oP 'Mean AP = \K\S+'`
    echo 'current accuracy: '$accuracy
    
    if [[ $max_accuracy < $accuracy ]]
    then
	max_accuracy=$accuracy
	echo 'max accuracy: '$max_accuracy
    fi
    
    # change GD policy when accuracy no big change
    if [[ $max_accuracy > $accuracy ]]
    then
	iter=`expr $iter + 1`
	if [[ $iter > $epoch ]]
	then
	    iter=0
	    cp models/pascal_voc/ZF/faster_rcnn_end2end_separable_conv/solver.prototxt models/pascal_voc/ZF/faster_rcnn_end2end_separable_conv/tmp.prototxt
	    awk -F ' ' '{
               if (($1 == "type:") && ($2 == "\"Adam\"")) {print $1 " " "\"SGD\"";}
               else if (($1 == "type:") && ($2 == "\"SGD\"")) {print $1 " " "\"Adam\"";}
               else {print $0;}
            }' models/pascal_voc/ZF/faster_rcnn_end2end_separable_conv/tmp.prototxt >| models/pascal_voc/ZF/faster_rcnn_end2end_separable_conv/solver.prototxt
	fi
    fi

    # save new caffemodel
    if [[ $accuracy > $svae_point ]]
    then
	cp output/faster_rcnn_end2end/voc_2007_trainval/Real_faster_rcnn.caffemodel output/faster_rcnn_end2end/voc_2007_trainval/"$accuracy""_""Real_faster_rcnn.caffemodel"
	cp output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_final.caffemodel output/faster_rcnn_end2end/voc_2007_trainval/"$accuracy""_""zf_faster_rcnn_final.caffemodel"
	cp faster_rcnn.log "$accuracy""_""faster_rcnn.log"
	echo "New caffemodel had been saved."
    fi

    # end train
    if [[ $accuracy > $end_point ]]
    then
	echo " ^-^ Train end."
	break
    fi
done