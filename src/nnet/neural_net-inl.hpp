#ifndef CXXNET_NNET_NEURAL_NET_INL_HPP_
#define CXXNET_NNET_NEURAL_NET_INL_HPP_
/*!
 * \file neural_net-inl.hpp
 * \brief implementation of common neuralnet
 * \author Tianqi Chen
 */
#include <vector>
#include <utility>
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include "../layer/layer.h"
#include "../layer/visitor.h"
#include "../updater/updater.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../utils/thread.h"
#include "./nnet_config.h"

namespace cxxnet {
namespace nnet {
/*! \brief implementation of abstract neural net */
template<typename xpu>
struct NeuralNet {
  /*! \brief network configuration configure */
  const NetConfig &cfg;
  /*! \brief maximum batch_size */
  mshadow::index_t max_batch;
  /*! \brief label information */
  layer::LabelInfo label_info;
  /*! \brief nodes in the neural net */
  std::vector<layer::Node<xpu> > nodes;
  /*! \brief extra nodes in neural net */
  std::vector<layer::Node<xpu> > extra_nodes;
  /*! \brief layers in the neural net */
  std::vector<layer::Connection<xpu> > connections;
  /*! \brief updaters in the neural net */
  std::vector<std::vector<updater::IAsyncUpdater<xpu>*> > updaters;
  /*! \brief random number generator */
  mshadow::Random<xpu> rnd;
  /*! \brief stream for this  */
  mshadow::Stream<xpu> *stream;
  /*! \brief net snapshots*/
  std::vector<NeuralNet<xpu>*> snapshots;
  /*! \brief trunk size for snapshots*/
  int trunk_size;
  // constructor do nothing
  NeuralNet(const NetConfig &cfg,
            mshadow::index_t batch_size,
            int seed,
            mshadow::Stream<xpu> *stream,
            int trunk_size=0)
      : cfg(cfg), rnd(seed), stream(stream), trunk_size(trunk_size) {
    // set maximum batch
    this->max_batch = batch_size;
    rnd.set_stream(stream);
    label_info.name2findex = &cfg.label_name_map;
    //snapshots.resize(trunk_size);
    //for (int i = 0; i < trunk_size; ++i) {
    //   NeuralNet<xpu> *pnet = new NeuralNet(cfg, batch_size, seed, stream, 0);
    //   snapshots[i] = pnet;
    //}
  }
  virtual ~NeuralNet(void) {
    this->FreeSpace();
  }
  /*! \brief save model to file */
  inline void SaveModel(utils::IStream &fo) const {
    for (index_t i = 0; i < connections.size(); ++i) {
      for (size_t j = 0; j < updaters[i].size(); ++j) {
        updaters[i][j]->UpdateWait();
      }
      if (connections[i].type != layer::kSharedLayer) {
        connections[i].layer->SaveModel(fo);
      }
    }
  }

  /*! \brief initial model parameters in the beginning */
  virtual void InitModel(bool flag = true) {
    this->InitNet(flag);
    this->ConfigConntions();
    this->SetModel();
  }
  inline void SetModel() {
    for (size_t i = 0; i < connections.size(); ++i) {
      if (this->cfg.layers[i].name != "") {
        utils::TrackerPrintf("Initializing layer: %s\n", this->cfg.layers[i].name.c_str());
      } else {
        utils::TrackerPrintf("Initializing layer: %d\n", static_cast<int>(i));
      }
      layer::Connection<xpu> &c = connections[i];
      if (c.layer) c.layer->InitConnection(c.nodes_in, c.nodes_out, &c.state);
      c.SetStream(stream);
    }
    for (size_t i = 0; i < connections.size(); ++i) {
      if (connections[i].type != layer::kSharedLayer) {
        if (connections[i].layer) connections[i].layer->InitModel();
      }
    }
  }
  /*! \brief load model from stream */
  inline void LoadModel(utils::IStream &fi, bool init_connection = true) {
    this->FreeSpace();
    this->InitNet();
    this->ConfigConntions();
    for (size_t i = 0; i < connections.size(); ++i) {
      if (connections[i].type != layer::kSharedLayer) {
        connections[i].SetStream(stream);
        connections[i].layer->LoadModel(fi);
      }
    }
    if (init_connection) {
      for (size_t i = 0; i < connections.size(); ++i) {
        layer::Connection<xpu> &c = connections[i];
        c.layer->InitConnection(c.nodes_in, c.nodes_out, &c.state);
        c.SetStream(stream);
      }
    }
  }
  /*!
   * \brief forward prop
   * \param is_train whether is training phase
   * \param batch the input batch
   */
  virtual void Forward(bool is_train,
                      mshadow::Tensor<cpu,4> batch,
                      std::vector<mshadow::Tensor<cpu,4> > extra_data,
                      bool need_sync, int t = -1, bool is_first_trunk=true) {
    // check if we need to adjust batch size according to the input
    std::vector<layer::Node<xpu> > &nd = t == -1 ? nodes : snapshots[t]->nodes;
    std::vector<layer::Connection<xpu> > &conn =  t == -1 ? connections : snapshots[t]->connections;
    this->AdjustBatchSize(batch.size(0));
    // copy data into node
    mshadow::Copy(nd[0].data, batch, stream);
    for (size_t i = 0; i < extra_data.size(); ++i) {
      mshadow::Copy(nd[i + 1].data, extra_data[i], stream);
    }
    // setup updater notification
    for (size_t i = conn.size(); i != 0; --i) {
      if (updaters.size() > 0) {
        for (size_t j = 0; j < updaters[i - 1].size(); ++j) {
          updaters[i - 1][j]->BeforeAllForward();
        }
      }
    }
    // start forward prop
    for (size_t i = 0; i < conn.size(); ++i) {
      layer::Connection<xpu> &c = conn[i];
      if (updaters.size() > 0) {
        for (size_t j = 0; j < updaters[i].size(); ++j) {
          updaters[i][j]->UpdateWait();
        }
      }
      c.layer->Forward(is_train, c.nodes_in, c.nodes_out, &c.state);
    }
  }
  /*!
   * \brief backprop
   * \param prop_to_input whether prop gradient to input node
   */
  inline void Backprop(bool prop_to_input,
                       bool need_update,
                       long update_epoch,
                       int t = -1, bool is_first_trunk = false) {
    std::vector<layer::Connection<xpu> > &conn =  t == -1 ? connections : snapshots[t]->connections;
    for (size_t i = conn.size(); i > 0; --i) {
      layer::Connection<xpu> &c = conn[i - 1];
      if (updaters.size() > 0) {
        for (size_t j = 0; j < updaters[i - 1].size(); ++j) {
          updaters[i - 1][j]->BeforeBackprop(c.nodes_in, c.nodes_out);
        }
      }
      c.layer->Backprop(i != 1 || prop_to_input,
                        c.nodes_in, c.nodes_out, &c.state);
      // wait backprop to complete before call update
      if (updaters.size() > 0) {
        if (updaters[i - 1].size() != 0) stream->Wait();
        for (size_t j = 0; j < updaters[i - 1].size(); ++j) {
          updaters[i - 1][j]->AfterBackprop(need_update, update_epoch);
        }
      }
    }
  }
  /*!
   * \brief explicitly synchronize the model parameters
   */
  inline void SyncParam(void) {
    // do a psedo update
    for (size_t i = connections.size(); i != 0; --i) {
      for (size_t j = 0; j < updaters[i - 1].size(); ++j) {
        updaters[i - 1][j]->BeforeAllForward();
      }
    }
    for (index_t i = 0; i < connections.size(); ++i) {
      for (size_t j = 0; j < updaters[i].size(); ++j) {
        updaters[i][j]->UpdateWait();
      }
    }
  }
  /*!
   * \brief update model parameters
   * \param epoch number of epoches
   */
  inline void Update(size_t epoch) {
    for (size_t i = 0; i < updaters.size(); ++ i) {
      for (size_t j = 0; j < updaters[i].size(); ++ j) {
        updaters[i][j]->Update(epoch);
      }
    }
  }
  /*!
   * \brief notify round start
   * \param round round counter
   */
  inline void StartRound(int round) {
    for (size_t i = 0; i < updaters.size(); ++ i) {
      for (size_t j = 0; j < updaters[i].size(); ++ j) {
        updaters[i][j]->StartRound(round);
      }
    }
  }
  // create the updaters
  inline void InitUpdaters(mshadow::ps::ISharedModel<xpu, real_t> *ps, int devid) {
    for (int i = 0; i < cfg.param.num_layers; ++i) {
      std::vector<updater::IAsyncUpdater<xpu>*> out;
      if (connections[i].type != layer::kSharedLayer) {
        updater::CreateAsyncUpdaters
            (i, devid, ps,
             cfg.updater_type.c_str(),
             &rnd, cfg.layers[i].type,
             connections[i].layer,
             &out);
        for (size_t k = 0; k < out.size(); ++k) {
          for (size_t j = 0; j < cfg.defcfg.size(); ++j) {
            out[k]->SetParam(cfg.defcfg[j].first.c_str(),
                             cfg.defcfg[j].second.c_str());
          }
          for (size_t j = 0; j < cfg.layercfg[i].size(); ++j) {
            out[k]->SetParam(cfg.layercfg[i][j].first.c_str(),
                             cfg.layercfg[i][j].second.c_str());
          }
          out[k]->SetStream(stream);
          out[k]->Init();
        }
      }
      updaters.push_back(out);
    }
    CHECK(updaters.size() == connections.size())
        << "updater size do not match number of layers";
  }
  // intialize the space of nodes
  inline void InitNodes(void) {
    for (size_t i = 0; i < nodes.size(); ++ i) {
      mshadow::Shape<4> s = nodes[i].data.shape_;
      nodes[i].AllocSpace();
      utils::TrackerPrintf("node[%s].shape: %u,%u,%u,%u\n",
         this->cfg.node_names[i].c_str(),
         s[0], s[1], s[2], s[3]);
    }
    for (size_t i = 0; i < extra_nodes.size(); ++i) {
      extra_nodes[i].AllocSpace();
    }
    for (size_t i = 0; i < snapshots.size(); ++i) {
      snapshots[i]->InitNodes();
    }
  }
  // intialize the neural net data structure
  virtual void InitNet(bool create_layer=true) {
    nodes.resize(cfg.param.num_nodes);
    mshadow::Shape<3> s = cfg.param.input_shape;
    // setup input shape
    nodes[0].data.shape_ = mshadow::Shape4(max_batch, s[0], s[1], s[2]);
    // setup extra data
    for (int i = 0; i < cfg.param.extra_data_num; ++i) {
      const std::vector<int>& extra_shape = cfg.extra_shape;
      nodes[i + 1].data.shape_ = mshadow::Shape4(
        max_batch, extra_shape[i * 3], extra_shape[i * 3 + 1], extra_shape[i * 3 + 2]);
    }
    // input layer
    for (int i = 0; i < cfg.param.num_layers; ++i) {
      const NetConfig::LayerInfo &info = cfg.layers[i];
      layer::Connection<xpu> c;
      c.type = info.type;
      for (size_t j = 0; j < info.nindex_in.size(); ++j) {
        c.nodes_in.push_back(&nodes[info.nindex_in[j]]);
      }
      for (size_t j = 0; j < info.nindex_out.size(); ++j) {
        c.nodes_out.push_back(&nodes[info.nindex_out[j]]);
      }
      if (c.type == layer::kSharedLayer) {
        CHECK(info.primary_layer_index >= 0) << "primary_layer_index problem";
        utils::Check(info.primary_layer_index < static_cast<int>(connections.size()),
                     "shared layer primary_layer_index exceed bound");
        c.layer = connections[info.primary_layer_index].layer;
        utils::Check(c.layer->AllowSharing(),
                     "some layer you set shared do not allow sharing");
      } else {
        if (create_layer) c.layer = layer::CreateLayer(c.type, &rnd, &label_info);
        else c.layer = NULL;
      }
      connections.push_back(c);
    }
  }
  // configure the parameters of layer
  inline void ConfigConntions(void) {
    for (int i = 0; i < cfg.param.num_layers; ++ i) {
      if (connections[i].type == layer::kSharedLayer) continue;
      for (size_t j = 0; j < cfg.defcfg.size(); ++j) {
        if (connections[i].layer == NULL) continue;
        connections[i].layer->SetParam(cfg.defcfg[j].first.c_str(),
                                       cfg.defcfg[j].second.c_str());
      }
      for (size_t j = 0; j < cfg.layercfg[i].size(); ++j) {
        if (connections[i].layer == NULL) continue;
        connections[i].layer->SetParam(cfg.layercfg[i][j].first.c_str(),
                                       cfg.layercfg[i][j].second.c_str());
      }
    }
  }
  // adjust batch size to a new value, the batch_size must be smaller than max_batch
  inline void AdjustBatchSize(mshadow::index_t batch_size) {
    CHECK(max_batch >= batch_size);
    if (batch_size != nodes[0].data.size(0)) {
      for (size_t i = 0; i < nodes.size(); ++i) {
        nodes[i].data.shape_[0] = batch_size;
      }
      for (size_t i = 0; i < extra_nodes.size(); ++i) {
        extra_nodes[i].data.shape_[0] = batch_size;
      }
      for (size_t i = 0; i < connections.size(); ++ i) {
        layer::Connection<xpu> &c = connections[i];
        c.layer->OnBatchSizeChanged(c.nodes_in, c.nodes_out, &c.state);
      }
      for (size_t i = 0; i < snapshots.size(); ++i) {
        snapshots[i]->AdjustBatchSize(batch_size);
      }
    }
  }
  /*! \brief clone label info from iterator */
  inline void CopyLabelInfo(int t, const layer::LabelInfo &info) {
    snapshots[t]->DeepCopyLabelInfo(info);
  }
  inline void DeepCopyLabelInfo(const layer::LabelInfo &info) {
    label_info.fields.resize(info.fields.size());
    for (size_t i = 0; i < label_info.fields.size(); ++i) {
      if (!(label_info.fields[i].label.shape_ == info.fields[i].label.shape_)) {
        label_info.fields[i].label.shape_ = info.fields[i].label.shape_;
        mshadow::AllocSpace(&label_info.fields[i].label, false);
      }
      mshadow::Copy(label_info.fields[i].label, info.fields[i].label, info.fields[i].label.stream_);
    }
    label_info.name2findex = info.name2findex;
  }
  /*! \brief free all space allocated in this struct*/
  inline void FreeSpace(void) {
    // wait all actions to complete before free
    stream->Wait();
    for (size_t i = 0; i < nodes.size(); ++i) {
      nodes[i].FreeSpace();
    }
    for (size_t i = 0; i < extra_nodes.size(); ++i) {
      extra_nodes[i].FreeSpace();
    }
    for (size_t i = 0; i < snapshots.size(); ++i) {
      snapshots[i]->FreeSpace();
    }

    for (size_t i = 0; i < connections.size(); ++i) {
      if (connections[i].type != layer::kSharedLayer) {
        delete connections[i].layer;
      }
    }
    for (size_t i = 0; i < updaters.size(); ++i) {
      for (size_t j = 0; j < updaters[i].size(); ++j) {
        delete updaters[i][j];
      }
    }
    nodes.clear();
    connections.clear();
    updaters.clear();
    snapshots.clear();
  }
};
}  // namespace nnet
}  // namespace cxxnet
#endif  // CXXNET_NNET_NEURAL_NET_INL_HPP_
