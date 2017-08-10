#include "caffe/quantized_layer.hpp"
#include "caffe/quantized_layer.cuh"

namespace caffe {


template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::Quantize_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  if (this->layer_param_.has_quantization_param()) {
    //LOG(INFO) << "Quantizing layer: " << this->layer_param_.name();
    const vector<shared_ptr<Blob > >& blobs = this->blobs();
    const QuantizationParameter& param = this->layer_param_.quantization_param();
    if (param.precision() != QuantizationParameter_Precision_FLOAT) {
      // Trim layer input
      for (int i = 0; i < std::min<int>(param.qparam_in_size(),bottom.size()); ++i) {
        if(param.qparam_in(i).quantize()) {
          this->QuantizeLayerInputs_gpu(bottom[i]->mutable_gpu_data<Ftype>(), i, bottom[i]->count());
        }
      }

      // Trim weights
      if(param.qparam_w().quantize() && blobs.size() > 0) {
        this->QuantizeWeights_gpu(blobs[0]->mutable_gpu_data<Ftype>(), blobs[0]->count(), param.rounding_scheme(), true);
        if (blobs.size() > 1) { //(this->bias_term_) {
          this->QuantizeWeights_gpu(blobs[1]->mutable_gpu_data<Ftype>(), blobs[1]->count(), param.rounding_scheme(), false);
        }
      }

      // Trim layer output
      if(param.qparam_out().quantize()) {
        for (int i = 0; i < top.size(); ++i) {
          this->QuantizeLayerOutputs_gpu(top[i]->mutable_gpu_data<Ftype>(), top[i]->count());
        }
      }
    }
  }
}


template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeWeights_gpu(Ftype* data, const int count, const int rounding, bool clip) {
  const QuantizationParameter& param = this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_w = param.qparam_w();
  switch (param.precision()) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    Trim2FixedPoint_gpu(data, count, param.power2_range(), qparam_w.bitwidth(),
        param.rounding_scheme(), qparam_w.fracbits(), qparam_w.scale(),
        qparam_w.offset(), qparam_w.unsigned_quant(), clip);
    break;
  case QuantizationParameter_Precision_FLOAT:
	  break;
  default:
    LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
    break;
  }
}

template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeLayerInputs_gpu(
    Ftype* data, const int blob_id, const int count) {
  const QuantizationParameter& param = this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_in = param.qparam_in(blob_id);
  switch (param.precision()) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_gpu(data, count, param.power2_range(), qparam_in.bitwidth(),
          param.rounding_scheme(), qparam_in.fracbits(), qparam_in.scale(),
          qparam_in.offset(), qparam_in.unsigned_quant(), true);
      break;
    case QuantizationParameter_Precision_FLOAT:
  	  break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
      break;
  }
}

template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::QuantizeLayerOutputs_gpu(Ftype* data,
      const int count) {
  const QuantizationParameter& param = this->layer_param_.quantization_param();
  const QuantizationParameter::QParams& qparam_out = param.qparam_out();
  switch (param.precision()) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_gpu(data, count, param.power2_range(), qparam_out.bitwidth(),
          param.rounding_scheme(), qparam_out.fracbits(), qparam_out.scale(),
          qparam_out.offset(), qparam_out.unsigned_quant(), true);
      break;
    case QuantizationParameter_Precision_FLOAT:
  	  break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << param.precision() << " for layer:" << this->layer_param_.name();
      break;
  }
}

template <typename Dtype>
__global__ void Trim2FixedPoint_kernel(Dtype* data, const int cnt,
    const int bitwidth, const int rounding, float scale, float inv_scale, float offset, float min_data, float max_data, bool clip) {
    CUDA_KERNEL_LOOP(index, cnt) {

    data[index] = data[index] * scale + offset;

    // Round data
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = rint(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = __float2int_rd(data[index] + RandUniform_device(index));
      break;
    default:
      break;
    }

#if CLIP_QUANT
    if(clip) {
        // Saturate data
        data[index] = (data[index]>(Dtype)max_data?
            (Dtype)max_data:(data[index]<(Dtype)min_data?(Dtype)min_data:data[index]));
    }
#endif

    data[index] = (data[index] - offset) * inv_scale;
  }
}


template<typename Ftype, typename Btype>
void QuantizedLayer<Ftype, Btype>::Trim2FixedPoint_gpu(Ftype* data, const int cnt, bool power2_range,
      const int bitwidth, const int rounding, int fracbits, float scale, float offset, bool unsigned_data, bool clip) {
  float inv_scale;
  if(power2_range) {
    scale = powf(2, fracbits);
    inv_scale = powf(2, -fracbits);
  } else {
    inv_scale = 1.0f/scale;
  }

  int qrange = unsigned_data? bitwidth :  (bitwidth - 1);
  float min_data = unsigned_data? 0 : -(powf(2, qrange));
  float max_data = +(powf(2, qrange) - 1);

  Trim2FixedPoint_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, bitwidth, rounding, scale, inv_scale, offset, min_data, max_data, clip);
}

template void QuantizedLayer<double, double>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<double, float>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<double, float16>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);

template void QuantizedLayer<float, double>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float, float>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float, float16>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);

template void QuantizedLayer<float16, double>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float16, float>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);
template void QuantizedLayer<float16, float16>::Quantize_gpu(const vector<Blob*>& bottom,const vector<Blob*>& top);


}  // namespace caffe


