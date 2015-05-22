#ifndef NEURAL_NET_LSTM_INL_HPP
#define NEURAL_NET_LSTM_INL_HPP
#pragma once

#include "./neural_net-inl.hpp"

namespace cxxnet {
namespace nnet {
template <typename xpu>
struct NeuralNetLSTM : public NeuralNet<xpu> {
  typedef NeuralNet<xpu> Parent;
public:
  NeuralNetLSTM(const NetConfig &cfg,
                mshadow::index_t batch_size,
                int seed,
                mshadow::Stream<xpu> *stream,
                int trunk_size=0) : NeuralNet<xpu>(cfg, batch_size, seed, stream, trunk_size) {
    Parent::snapshots.resize(trunk_size);
    for (int i = 0; i < trunk_size; ++i) {
      NeuralNet<xpu> *pnet = new NeuralNetLSTM(cfg, batch_size, seed, stream, 0);
      Parent::snapshots[i] = pnet;
    }
  }
  virtual void Forward(bool is_train,
                       mshadow::Tensor<cpu,4> batch,
                       std::vector<mshadow::Tensor<cpu,4> > extra_data,
                       bool need_sync, int t = -1, bool is_first_trunk=true) {
    this->PrepForward(t, is_first_trunk);
    if (t >= 0) {
      Parent::snapshots[t]->Forward(is_train, batch, extra_data, need_sync);
    } else {
      Parent::Forward(is_train, batch, extra_data, need_sync);
    }
  }
  virtual void Backprop(bool prop_to_input,
                        bool need_update,
                        long update_epoch,
                        int t = -1, bool is_first_trunk = false) {
    this->PrepBackprop(t);
    if (t >= 0) {
      Parent::snapshots[t]->Backprop(prop_to_input, need_update, update_epoch);
    } else {
      Parent::Backprop(prop_to_input, need_update, update_epoch);
    }
  }
  virtual void InitNet(bool create_layer=true) {
    Parent::nodes.resize(Parent::cfg.param.num_nodes);
    mshadow::Shape<3> s = Parent::cfg.param.input_shape;
    // setup lstm nodes number
    int lstm_number = 0;
    int extra_nodes_number = 0;
    for (int i = 0; i < Parent::cfg.param.num_layers; ++i) {
      const NetConfig::LayerInfo &info = Parent::cfg.layers[i];
      if (info.type == layer::kLSTM) lstm_number++;
    }
    extra_nodes_number = Parent::snapshots.size() > 0 ? lstm_number * 7 : lstm_number;
    Parent::extra_nodes.resize(extra_nodes_number);
    lstm_number = 0;
    // setup input shape
    Parent::nodes[0].data.shape_ = mshadow::Shape4(Parent::max_batch, s[0], s[1], s[2]);
    // setup extra data
    for (int i = 0; i < Parent::cfg.param.extra_data_num; ++i) {
      const std::vector<int>& extra_shape = Parent::cfg.extra_shape;
      Parent::nodes[i + 1].data.shape_ = mshadow::Shape4(
        Parent::max_batch, extra_shape[i * 3], extra_shape[i * 3 + 1], extra_shape[i * 3 + 2]);
    }
    // input layer
    for (int i = 0; i < Parent::cfg.param.num_layers; ++i) {
      const NetConfig::LayerInfo &info = Parent::cfg.layers[i];
      layer::Connection<xpu> c;
      c.type = info.type;
      for (size_t j = 0; j < info.nindex_in.size(); ++j) {
        c.nodes_in.push_back(&Parent::nodes[info.nindex_in[j]]);
      }
      for (size_t j = 0; j < info.nindex_out.size(); ++j) {
        c.nodes_out.push_back(&Parent::nodes[info.nindex_out[j]]);
      }
      if (c.type == layer::kLSTM) {
        c.nodes_in.resize(3);
        c.nodes_out.resize(7);
        c.nodes_out[1] = &(this->Parent::extra_nodes[lstm_number++]);
        for (size_t  d = 2; d < 7; ++d) {
          if (create_layer) {
            c.nodes_out[d] = &(this->Parent::extra_nodes[lstm_number++]);
          } else {
            c.nodes_out[d] = NULL;
          }
        }
      }
      if (!create_layer) {
        c.layer = NULL;
      }
      if (c.type == layer::kSharedLayer ) {
        CHECK(info.primary_layer_index >= 0) << "primary_layer_index problem";
        utils::Check(info.primary_layer_index < static_cast<int>(Parent::connections.size()),
                     "shared layer primary_layer_index exceed bound");
        c.layer = Parent::connections[info.primary_layer_index].layer;
        utils::Check(c.layer->AllowSharing(),
                     "some layer you set shared do not allow sharing");
      } else {
        c.layer = layer::CreateLayer(c.type, &(this->Parent::rnd), &(this->Parent::label_info));
      }
      if (c.type == layer::kSoftmax || c.type == layer::kLpLoss) {
        if (c.layer == NULL) c.layer = layer::CreateLayer(c.type, &(this->Parent::rnd), &(this->Parent::label_info));
      }
      Parent::connections.push_back(c);
    }
  }
  virtual void InitModel(bool flag = true) {
    this->InitNet(flag);
    Parent::ConfigConntions();
    Parent::SetModel();
    for (size_t i = 0; i < Parent::snapshots.size(); ++i) {
      NeuralNetLSTM *pnet = static_cast<NeuralNetLSTM*>(Parent::snapshots[i]);
      pnet->InitModel(false);
    }
  }
private:
  void PrepForward(int t, bool is_first) {
    if (t < 0) return;
    if (is_first) {
      for (size_t i = 0; i < Parent::connections.size(); ++i) {
        Parent::connections[i].nodes_out[0]->data = 0.0f;
        if (Parent::connections[i].type == layer::kLSTM) {
          Parent::connections[i].nodes_out[1]->data = 0.0f;
        }
      }
    }
    if (t == 0 && is_first == false) {
      for (size_t i = 0; i < Parent::connections.size(); ++i) {
        if (Parent::connections[i].type == layer::kLSTM) {
          mshadow::Copy(Parent::connections[i].nodes_out[0]->data,
                        Parent::snapshots[Parent::trunk_size - 1]->connections[i].nodes_out[0]->data,
                        Parent::connections[i].nodes_out[0]->data.stream_);
          mshadow::Copy(Parent::connections[i].nodes_out[1]->data,
                        Parent::snapshots[Parent::trunk_size - 1]->connections[i].nodes_out[1]->data,
                        Parent::connections[i].nodes_out[0]->data.stream_);
        }
      }
    }
    for (size_t i = 0; i < Parent::snapshots[t]->connections.size(); ++i) {
      layer::Connection<xpu> &now_c  = Parent::snapshots[t]->connections[i];
      layer::Connection<xpu> &last_c = t == 0 ? Parent::connections[i] : Parent::snapshots[t - 1]->connections[i];
      if (now_c.layer == NULL) now_c.layer = last_c.layer;
      if (now_c.type == layer::kLSTM) {
        now_c.nodes_in[1] = last_c.nodes_out[0];
        now_c.nodes_in[2] = last_c.nodes_out[1];
      }
    }
  }
  void PrepBackprop(int t) {
    if (t < 0) return;
    if (t == Parent::trunk_size - 1) {
      for (size_t i = 0; i < Parent::connections.size(); ++i) {
        if (Parent::connections[i].type == layer::kLSTM) {
          for (int j = 2; j < 7; ++j) {
            Parent::connections[i].nodes_out[j]->data = 0.0f;
          }
        }
      }
    }
    for (size_t i = 0; i < Parent::snapshots[t]->connections.size(); ++i) {
      layer::Connection<xpu> &now_c  = Parent::snapshots[t]->connections[i];
      layer::Connection<xpu> &last_c = t == Parent::trunk_size - 1 ? Parent::connections[i] : Parent::snapshots[t + 1]->connections[i];
      if (now_c.type == layer::kLSTM) {
        for (int j = 2; j < 7; ++j) {
          now_c.nodes_out[j] = last_c.nodes_out[j];
        }
      }
    }
  }
}; // class NeuralNetLSTM
} // namespace nnet
} // namespace cxxnet
#endif // NEURAL_NET_LSTM_INL_HPP
