# NBFNet-Attention
This is the Part-C project for Graph Representation Learning, based on code https://github.com/DeepGraphLearning/NBFNet

There are 3 variants of attention, i.e. Bilinear,GAT, and GAT-Sep

Equation for NBFNet-Attention:

$$
\mathbf{h}_{\mathbf{q}}^{(0)}(u,v) \leftarrow \mathbf{1}_{(u=v)} \cdot \mathbf{q}
$$

$$
    \mathbf{h}_{\mathbf{q}}^{(t)}(u,v) \leftarrow \sigma\bigl(
    \mathbf{h}_{\mathbf{q}}^{(0)}(u,v) \cdot \mathbf{W}_{1}^{(t)} 
    + \Bigl(
        \sum_{\{w \mid (w,r,v) \in \mathcal{E}(v)\}} \alpha_{u,\mathbf{q}}^{(t-1)}(v,r,w) \cdot \mathbf{h}_{\mathbf{q}}^{(t-1)}(u,w) \cdot \mathbf{w_\mathbf{q}}(w,r,v)
    \Bigr) \cdot \mathbf{W}_2^{(t)}
\bigr) 
$$

where

$$
\alpha_{u,\mathbf{q}}^{(t-1)}(v,r,w) = \frac{\exp{(e_{u,\mathbf{q}}^{(t-1)}(v,r,w)})}{\sum_{\{w^{\prime} \mid (w^{\prime},r^{\prime},v) \in \mathcal{E}(v)\}}\exp{(e_{u,\mathbf{q}}^{(t-1)}(v,r^{\prime},w^{\prime}))})}\nonumber 
$$

Attention-Bilinear:

$$
e_{u,\mathbf{q}}^{(t-1)}(v,r,w) =  (\mathbf{h}_{\mathbf{q}}^{(t-1)}(u,v))^\intercal \mathbf{W}^{(t-1)}_{atten} (\mathbf{h}_{\mathbf{q}}^{(t-1)}(u,w) \cdot \mathbf{w_\mathbf{q}}(w,r,v))
$$

Attention-GAT:

$$
e_{u,\mathbf{q}}^{(t-1)}(v,r,w) = \operatorname{LeakyReLU}(\mathbf{a}^\intercal[ \mathbf{W}^{(t-1)}_{atten} \mathbf{h}_{\mathbf{q}}^{(t-1)}(u,v) \mid\mid \mathbf{W}^{(t-1)}_{atten} (\mathbf{h}_{\mathbf{q}}^{(t-1)}(u,w) \cdot \mathbf{w_\mathbf{q}}(w,r,v))])
$$

Attention-GAT-Sep:

$$
e_{u,\mathbf{q}}^{(t-1)}(v,r,w) = \operatorname{LeakyReLU}(\mathbf{a}^\intercal[ \mathbf{W}^{(t-1)}_{atten} \mathbf{h}_{\mathbf{q}}^{(t-1)}(u,v) \mid\mid \mathbf{W}^{(t-1)}_{atten} \mathbf{h}_{\mathbf{q}}^{(t-1)}(u,w) \mid\mid \mathbf{W}^{(t-1)}_{rel}\mathbf{w_\mathbf{q}}(w,r,v)])
$$

To run the code for each attention variant, just change the config file .yaml.  
      Attention-mode = Bilinear/GAT/GAT_Sep, and num_head indicates the number of head. 
      
Note that we only support num_head=1 for Bilinear Attention-mode.

Also, to make attention works, we need to set $\texttt{AGGREGATE}$ = sum

To run the code, see https://github.com/HxyScotthuang/NBFNet-Attention/blob/main/Original_ReadMe.md
