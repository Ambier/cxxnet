#ifndef LSTM_LAYER_INL_HPP
#define LSTM_LAYER_INL_HPP
#pragma once

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "./op.h"

namespace cxxnet {
namespace layer {

template<typename xpu,
         typename ZOp, typename ZOpGrad,
         typename IOp, typename IOpGrad,
         typename FOp, typename FOpGrad,
         typename OOp, typename OOpGrad,
         typename COp, typename COpGrad>
class LSTMLayer : public ILayer<xpu> {
  public:
    LSTMLayer(mshadow::Random<xpu> *p_rnd) : prnd_(p_rnd) {}
    virtual ~LSTMLayer() {}
    virtual void SetParam(const char *name, const char *val) {
      param_.SetParam(name, val);
    }
    virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                                const std::vector<Node<xpu>*> &nodes_out,
                                ConnectState<xpu> *p_cstate) {
      // input: x_t, y_{t-1}, c_{t-1}
      // output: y_t, c_t, delta_z_{t+1}, delta_i_{t+1}, delta_o_{t+1}, delta_f_{t+1}, delta_c_{t+1}
      // state: z_no_act, i_no_act, f_no_act, o_no_act, c_t_no_act, y_back
      // Or we can save the act to speed up?
      const int batch_size = nodes_in[0]->data.shape_[0];
      mshadow::Shape<4> oshape = mshadow::Shape4(batch_size, 1, 1, param_.num_hidden);
      if (param_.num_input_node == 0) {
        param_.num_input_node = static_cast<int>(nodes_in[0]->data.size(3));
      }
      for (size_t i = 0; i < nodes_out.size(); ++i) {
        if (nodes_out[i]) nodes_out[i]->data.shape_ = oshape;
      }
      p_cstate->states.resize(6);
      for (size_t i = 0; i < p_cstate->states.size(); ++i) {
        p_cstate->states[i].Resize(oshape);
      }
    }
    virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
      pvisitor->Visit("wmat", W_, gW_);
      pvisitor->Visit("umat", U_, gU_);
      if (param_.no_bias == 0) {
        pvisitor->Visit("bias", b_, gb_);
      }
    }
    virtual void InitModel() {
      W_.Resize(mshadow::Shape3(4, param_.num_hidden, param_.num_input_node));
      U_.Resize(mshadow::Shape3(4, param_.num_hidden, param_.num_hidden));
      b_.Resize(mshadow::Shape2(2, param_.num_hidden));
      for (int i = 0; i < 4; ++i) {
        mshadow::Tensor<xpu, 2> wmat = W_[i];
        mshadow::Tensor<xpu, 2> umat = U_[i];
        param_.RandInitWeight(this->prnd_, wmat, wmat.size(1), wmat.size(0));
        param_.RandInitWeight(this->prnd_, umat, umat.size(1), umat.size(0));
      }
      b_ = param_.init_bias;
      gW_.Resize(W_.shape_);
      gU_.Resize(U_.shape_);
      gb_.Resize(b_.shape_);
      gW_ = 0.0f;
      gU_ = 0.0f;
      gb_ = 0.0f;
    }
    virtual void SaveModel(utils::IStream &fo) const {
      fo.Write(&param_, sizeof(LayerParam));
      W_.SaveBinary(fo);
      U_.SaveBinary(fo);
      b_.SaveBinary(fo);
    }
    virtual void LoadModel(utils::IStream &fi) {
      utils::Check(fi.Read(&param_, sizeof(LayerParam)) != 0,
                   "FullConnectLayer:LoadModel invalid model file");
      W_.LoadBinary(fi);
      U_.LoadBinary(fi);
      b_.LoadBinary(fi);
      gW_.Resize(W_.shape_);
      gU_.Resize(U_.shape_);
      gb_.Resize(b_.shape_);
      gW_ = 0.0f;
      gU_ = 0.0f;
      gb_ = 0.0f;

    }
    virtual void SetStream(mshadow::Stream<xpu> *stream) {
      W_.set_stream(stream);
      U_.set_stream(stream);
      b_.set_stream(stream);
      gW_.set_stream(stream);
      gU_.set_stream(stream);
      gb_.set_stream(stream);
    }
    virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
      mshadow::Tensor<xpu, 2> Wi_ = W_[0];
      mshadow::Tensor<xpu, 2> Wf_ = W_[1];
      mshadow::Tensor<xpu, 2> Wo_ = W_[2];
      mshadow::Tensor<xpu, 2> Wz_ = W_[3];

      mshadow::Tensor<xpu, 2> Ui_ = U_[0];
      mshadow::Tensor<xpu, 2> Uf_ = U_[1];
      mshadow::Tensor<xpu, 2> Uo_ = U_[2];
      mshadow::Tensor<xpu, 2> Uz_ = U_[3];

      mshadow::Tensor<xpu, 1> bi_ = b_[0];
      mshadow::Tensor<xpu, 1> bf_ = b_[1];
      mshadow::Tensor<xpu, 1> bo_ = b_[2];
      mshadow::Tensor<xpu, 1> bz_ = b_[3];

      mshadow::Tensor<xpu, 2> x = nodes_in[0]->mat();
      mshadow::Tensor<xpu, 2> last_y = nodes_in[1]->mat();
      mshadow::Tensor<xpu, 2> last_c = nodes_in[2]->mat();
      mshadow::Tensor<xpu, 2> y = nodes_out[0]->mat();
      mshadow::Tensor<xpu, 2> c_t = nodes_out[1]->mat();

      mshadow::Tensor<xpu, 2> z = p_cstate->states[0].FlatTo2D();
      mshadow::Tensor<xpu, 2> i = p_cstate->states[1].FlatTo2D();
      mshadow::Tensor<xpu, 2> f = p_cstate->states[2].FlatTo2D();
      mshadow::Tensor<xpu, 2> o = p_cstate->states[3].FlatTo2D();
      mshadow::Tensor<xpu, 2> c = p_cstate->states[4].FlatTo2D();
      mshadow::Tensor<xpu, 2> y_back = p_cstate->states[5].FlatTo2D();
      index_t nbatch = x.size(0);
      using namespace mshadow::expr;
      z = dot(x, Wz_.T());
      z += dot(last_y, Uz_.T());
      i = dot(x, Wi_.T());
      i += dot(last_y, Ui_.T());
      f = dot(x, Wf_.T());
      f += dot(last_y, Uf_.T());
      o = dot(x, Wo_.T());
      o += dot(last_y, Uo_.T());
      if (param_.no_bias == 0) {
        z += repmat(bz_, nbatch);
        i += repmat(bi_, nbatch);
        f += repmat(bf_, nbatch);
        o += repmat(bo_, nbatch);
      }
      c = F<ZOp>(z) * F<IOp>(i) + last_c * F<FOp>(f);
      c_t = F<COp>(c);
      y = c_t * F<OOp>(o);
      mshadow::Copy(y_back, y, y_back.stream_);
    }

    virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
      using namespace mshadow::expr;

      mshadow::Tensor<xpu, 2> Wi_ = W_[0];
      mshadow::Tensor<xpu, 2> Wf_ = W_[1];
      mshadow::Tensor<xpu, 2> Wo_ = W_[2];
      mshadow::Tensor<xpu, 2> Wz_ = W_[3];

      mshadow::Tensor<xpu, 2> Ui_ = U_[0];
      mshadow::Tensor<xpu, 2> Uf_ = U_[1];
      mshadow::Tensor<xpu, 2> Uo_ = U_[2];
      mshadow::Tensor<xpu, 2> Uz_ = U_[3];

      mshadow::Tensor<xpu, 2> gWi_ = gW_[0];
      mshadow::Tensor<xpu, 2> gWf_ = gW_[1];
      mshadow::Tensor<xpu, 2> gWo_ = gW_[2];
      mshadow::Tensor<xpu, 2> gWz_ = gW_[3];

      mshadow::Tensor<xpu, 2> gUi_ = gU_[0];
      mshadow::Tensor<xpu, 2> gUf_ = gU_[1];
      mshadow::Tensor<xpu, 2> gUo_ = gU_[2];
      mshadow::Tensor<xpu, 2> gUz_ = gU_[3];

      mshadow::Tensor<xpu, 1> gbi_ = gb_[0];
      mshadow::Tensor<xpu, 1> gbf_ = gb_[1];
      mshadow::Tensor<xpu, 1> gbo_ = gb_[2];
      mshadow::Tensor<xpu, 1> gbz_ = gb_[3];

      mshadow::Tensor<xpu, 2> delta_y = nodes_out[0]->mat();
      mshadow::Tensor<xpu, 2> delta_z = nodes_out[2]->mat();
      mshadow::Tensor<xpu, 2> delta_i = nodes_out[3]->mat();
      mshadow::Tensor<xpu, 2> delta_o = nodes_out[4]->mat();
      mshadow::Tensor<xpu, 2> delta_f = nodes_out[5]->mat();
      mshadow::Tensor<xpu, 2> delta_c = nodes_out[6]->mat();

      mshadow::Tensor<xpu, 2> x = nodes_in[0]->mat();
      mshadow::Tensor<xpu, 2> last_c = nodes_in[2]->mat();

      mshadow::Tensor<xpu, 2> z = p_cstate->states[0].FlatTo2D();
      mshadow::Tensor<xpu, 2> i = p_cstate->states[1].FlatTo2D();
      mshadow::Tensor<xpu, 2> f = p_cstate->states[2].FlatTo2D();
      mshadow::Tensor<xpu, 2> o = p_cstate->states[3].FlatTo2D();
      mshadow::Tensor<xpu, 2> c = p_cstate->states[4].FlatTo2D();
      mshadow::Tensor<xpu, 2> y_back = p_cstate->states[5].FlatTo2D();

      gUi_ += dot(delta_i.T(), y_back);
      gUf_ += dot(delta_f.T(), y_back);
      gUz_ += dot(delta_z.T(), y_back);
      gUo_ += dot(delta_o.T(), y_back);

      delta_y += dot(delta_z, Uz_);
      delta_y += dot(delta_i, Ui_);
      delta_y += dot(delta_f, Uf_);
      delta_y += dot(delta_o, Uo_);

      delta_o = delta_y * F<COp>(c) * F<OOpGrad>(o);
      delta_c = delta_y * F<OOp>(o) * F<COpGrad>(c) + delta_c * delta_f;
      delta_f = delta_c * last_c * F<FOpGrad>(f);
      delta_i = delta_c * F<ZOp>(z) * F<IOpGrad>(i);
      delta_z = delta_c * F<IOp>(i) * F<ZOpGrad>(z);

      gWi_ += dot(delta_i.T(), x);
      gWz_ += dot(delta_z.T(), x);
      gWf_ += dot(delta_f.T(), x);
      gWo_ += dot(delta_o.T(), x);

      x = dot(delta_z, Wz_);
      x += dot(delta_i, Wi_);
      x += dot(delta_f, Wf_);
      x += dot(delta_o, Wo_);

      if (param_.no_bias == 0) {
        gbi_ += sum_rows(delta_i);
        gbo_ += sum_rows(delta_o);
        gbf_ += sum_rows(delta_f);
        gbz_ += sum_rows(delta_z);
      }
    }

  protected:
    /*! \brief parameters for LSTM */
    LayerParam param_;
    /*! \brief random number generator */
    mshadow::Random<xpu> *prnd_;
    /*! \brief weight */
    mshadow::TensorContainer<xpu, 3> W_;
    mshadow::TensorContainer<xpu, 3> U_;
    /*! \brief bias */
    mshadow::TensorContainer<xpu, 2> b_;
    /*! \brief weight gradient */
    mshadow::TensorContainer<xpu, 3> gW_;
    mshadow::TensorContainer<xpu, 3> gU_;
    /*! \brief bias gradient */
    mshadow::TensorContainer<xpu, 2> gb_;
}; // class LSTMLayer

} // namespace layer
} // namespace cxxnet
#endif // LSTM_LAYER_INL_HPP
