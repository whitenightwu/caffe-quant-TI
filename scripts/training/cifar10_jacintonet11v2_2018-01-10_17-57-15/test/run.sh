cd ../../caffe-jacinto//build/tools/caffe.bin
../../caffe-jacinto//build/tools/caffe.bin train \
--solver="training/cifar10_jacintonet11v2_2018-01-10_17-57-15/test/solver.prototxt" \
--weights="training/cifar10_jacintonet11v2_2018-01-10_17-57-15/sparse/cifar10_jacintonet11v2_iter_64000.caffemodel" \
--gpu "1" 2>&1 | tee training/cifar10_jacintonet11v2_2018-01-10_17-57-15/test/run.log
