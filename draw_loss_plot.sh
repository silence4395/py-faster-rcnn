./caffe-fast-rcnn/tools/extra/faster_rcnn_parse_log.sh faster_rcnn.log_1
cp faster_rcnn.log_1.train faster_rcnn.log.train
gnuplot caffe-fast-rcnn/tools/extra/faster_plot_log.gnuplot
eog your_chart_name.png&
