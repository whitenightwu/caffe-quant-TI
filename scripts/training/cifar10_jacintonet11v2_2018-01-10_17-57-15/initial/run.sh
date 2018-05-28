cd ../../caffe-jacinto//build/tools/caffe.bin
../../caffe-jacinto//build/tools/caffe.bin train \
--solver="training/cifar10_jacintonet11v2_2018-01-10_17-57-15/initial/solver.prototxt" \
--gpu "1" 2>&1 | tee training/cifar10_jacintonet11v2_2018-01-10_17-57-15/initial/run.log
