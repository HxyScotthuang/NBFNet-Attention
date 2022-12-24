import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers
from torchdrug.layers import functional
import functools
class GeneralizedRelationalConv(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True,attention_mode = "Bilinear",num_head=1, negative_slope=0.2):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.attention_mode = attention_mode
        self.num_head = num_head
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)
        self.attention_linear = nn.Linear(input_dim,output_dim) # Attention matrix W_atten for Bilinear/ GAT
        if self.attention_mode == "GAT_sep":
          self.attention_linear_2 = nn.Linear(input_dim,output_dim) # Relation matrix for Attention W_rel
          self.attention_query_sep = nn.Parameter(torch.zeros(num_head, output_dim * 3//num_head)) # For GAT_sep
        else:
          self.attention_query = nn.Parameter(torch.zeros(num_head, output_dim * 2//num_head)) # For GAT
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope) # For GAT
        


    def message(self, graph, input):
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        relation_input = relation_input.transpose(0, 1)
        node_input = input[node_in]
        edge_input = relation_input[relation]

        if self.message_func == "transe":
            message = edge_input + node_input
        elif self.message_func == "distmult":
            message = edge_input * node_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        # We add in the attention mechanism: we compute a_{i,j}^(t-1) for each message h(u,w)^(t-1)*w_q(w,r,v) with h(u,v)^(t-1)
        
        # Bilinear  m(v,r,w) = exp(h(v)^T * W * (h(w)*wq(w,r,v) ))
        # GAT    m(v,r,w) = exp(a^T[W * h(v) || W * (h(w)*wq(w,r,v)] ))
        # GAT_sep  m(v,r,w)= exp(a^T[W * h(v) || W * (h(w)|| W_2 * wq(w,r,v)] ))
        
        #message = h(w)*wq*(w,r,v)
        if self.attention_mode == "Bilinear":
          key = self.attention_linear(message)[node_out] # key = W*(h(w)*wq*(w,r,v))
          weight = torch.einsum("nhd, nhd -> nh", input[node_in], key)
        elif self.attention_mode == "GAT":
           key = torch.stack([self.attention_linear(input)[node_in], self.attention_linear(message)[node_out]], dim=-1)
           key = key.view(-1,batch_size, *self.attention_query.shape)
           weight = torch.einsum("hd, nbhd -> nh",self.attention_query, key) #consider num_head
           weight = self.leaky_relu(weight)
        elif self.attention_mode == "GAT_sep":
           key = torch.stack([self.attention_linear(input)[node_in], self.attention_linear(input)[node_out],self.attention_linear_2(edge_input)], dim=-1)
           key = key.view(-1,batch_size, *self.attention_query_sep.shape)
           weight = torch.einsum("hd, nbhd -> nh",self.attention_query_sep, key) #consider num_head
           weight = self.leaky_relu(weight)
        else:
          raise NotImplementedError
        weight = weight - scatter_max(weight, node_out, dim=0, dim_size=graph.num_node)[0][node_out] #normalized out,avoid nan error
        attention = weight.exp()
        normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out] # denominator
        attention = attention / (normalizer + self.eps) # final attention
        if self.attention_mode == "Bilinear":
          attention = attention.unsqueeze(-1).expand_as(message)
          final_message = (attention * message)
        elif self.attention_mode == "GAT":
          value = message.view(-1,batch_size, self.num_head, self.attention_query.shape[-1] // 2)
          attention = attention.unsqueeze(1).unsqueeze(-1).expand_as(value)
          final_message = (attention * value).flatten(-2)
        elif self.attention_mode == "GAT_sep":
          value = message.view(-1,batch_size, self.num_head, self.attention_query_sep.shape[-1] // 3)
          attention = attention.unsqueeze(1).unsqueeze(-1).expand_as(value)
          final_message = (attention * value).flatten(-2)  
        else:
          raise NotImplementedError
        

        message = torch.cat([final_message, graph.boundary])

        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update

        '''
        # Uncomment this for original NBFNet without attention. 
    def message_and_aggregate(self, graph, input):
        if graph.requires_grad or self.message_func == "rotate":
            return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        input = input.flatten(1)
        boundary = graph.boundary.flatten(1)

        degree_out = graph.degree_out.unsqueeze(-1) + 1
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)
        else:
            relation_input = self.relation.weight.repeat(1, batch_size)
        adjacency = graph.adjacency.transpose(0, 1)

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update.view(len(update), batch_size, -1)
        '''
    def combine(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output