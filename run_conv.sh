#echo "Begin CONV pruning..."
#./experiments/scripts/faster_rcnn_alt_opt.sh 1 ZF pascal_voc --prun 3 |& tee faster_rcnn.log
#echo "Begin FC pruning..."
#./experiments/scripts/faster_rcnn_alt_opt.sh 1 ZF pascal_voc --prun 4 |& tee faster_rcnn.log
#echo "Begin CONV retrain..."
#./experiments/scripts/faster_rcnn_alt_opt.sh 1 ZF pascal_voc --prun 1 |& tee faster_rcnn.log
echo "Begin alternate conv and fc retrain..."
./experiments/scripts/faster_rcnn_alt_opt.sh 1 ZF pascal_voc --prun 0 |& tee faster_rcnn.log