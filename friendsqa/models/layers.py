
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Tuple, Union


class GATModel(nn.Module):
    def __init__(self, config, args, node_dict, edge_dict, all_edges):
        super().__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_inp = self.n_hid = self.n_out = config.hidden_size
        self.n_layers = args.num_gat_layers
        self.adapt_ws = nn.ModuleList([nn.Linear(self.n_inp, self.n_hid) for _ in range(max(node_dict.values())+1)])
        self.gcs = nn.ModuleList([GATLayer(self.n_hid, self.n_hid, node_dict, edge_dict, all_edges, args.num_gat_heads) for _ in range(self.n_layers)])
        self.out = nn.ModuleList([nn.Linear(self.n_hid, self.n_out) for _ in range(max(node_dict.values())+1)])

    def forward(self, G, feats):
        h = {}
        for ntype in self.node_dict:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](feats[ntype]))
        for i in range(self.n_layers):
            h, utterance_to_utterance_matrix, utterance_to_question_matrix, question_to_utterance_matrix = self.gcs[i](G, h)

        out = {}
        for ntype in self.node_dict:
            n_id = self.node_dict[ntype]
            out[ntype] = self.out[n_id](h[ntype])
        return out, utterance_to_utterance_matrix, utterance_to_question_matrix, question_to_utterance_matrix

class GATLayer(nn.Module):
    def __init__(self,
                in_dim,
                out_dim,
                node_dict,
                edge_dict,
                all_edges,
                n_heads,
                dropout=0.2,
                feat_drop=0.,
                attn_drop=0.,
                negative_slope=0.2):
        super(GATLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.all_edges = all_edges
        self.num_types = max(node_dict.values())+1
        self.num_relations = max(edge_dict.values())+1
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.src_linears = nn.ModuleList()
        self.dst_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for t in range(self.num_types):
            self.src_linears.append(nn.Linear(in_dim,   out_dim))
            self.dst_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations,self.n_heads))
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(self.num_relations,n_heads,self.d_k)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(self.num_relations,n_heads,self.d_k)))
        
        self.feat_drop= nn.Dropout(feat_drop)
        self.attn_drop=nn.Dropout(attn_drop)
        self.leaky_relu=nn.LeakyReLU(negative_slope)
        
        self.skip=nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)
        
        self.cross_edge_weights = {}
        self.dst_to_cross_edge_id = {}
        dst_nodes_num = {}
        for e in all_edges:
            if e[-1] not in dst_nodes_num:
                dst_nodes_num[e[-1]] = 1
                self.dst_to_cross_edge_id[(e[1],e[-1])] = 0
            else:
                self.dst_to_cross_edge_id[(e[1],e[-1])] = dst_nodes_num[e[-1]]
                dst_nodes_num[e[-1]] = dst_nodes_num[e[-1]] + 1
        # print(dst_nodes_num)
        for node, num in dst_nodes_num.items():
            self.cross_edge_weights[node] = nn.Parameter(torch.FloatTensor(size=(num,)))
        
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        nn.init.xavier_uniform_(self.relation_pri)
        
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.d_k)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, G, h):
        
        cross_edge_weights = {}
        for node, rep in self.cross_edge_weights.items():
            cross_edge_weights[node] = F.softmax(rep)

        utterance_to_utterance_matrix, utterance_to_question_matrix, question_to_utterance_matrix=None, None, None
        updated_h = {}
        for triple in self.all_edges:
            
            srctype, etype, dsttype = triple
            
            src_linear = self.src_linears[self.node_dict[srctype]]
            dst_linear = self.dst_linears[self.node_dict[dsttype]]
                
            feat_src = self.transpose_for_scores(src_linear(self.feat_drop(h[srctype])))  # batch, n_heads, node_num1, hidden
            feat_dst = self.transpose_for_scores(dst_linear(self.feat_drop(h[dsttype])))  # batch, n_heads, node_num2, hidden
            
            e_id = self.edge_dict[etype]
            el = (feat_src * self.attn_l[e_id].unsqueeze(0).unsqueeze(2)).sum(dim=-1) # batch, n_heads, node_num1
            er = (feat_dst * self.attn_r[e_id].unsqueeze(0).unsqueeze(2)).sum(dim=-1) # batch, n_heads, node_num2
            
            attn_score = er.unsqueeze(-1) + el.unsqueeze(2)
            attn_score = self.leaky_relu(attn_score + self.relation_pri[e_id].unsqueeze(0).unsqueeze(2).unsqueeze(3))
            
            if next(self.parameters()).dtype == torch.float16:
                zero_vec = -65500 * torch.ones_like(attn_score)
            else:
                zero_vec = -1e30 * torch.ones_like(attn_score)
            
            adj = G[triple].permute(0,2,1)  # node_num2, node_num1
            adj = adj.unsqueeze(1).expand_as(attn_score)
            attn_scoree = torch.where(adj > 0, attn_score, zero_vec.to(attn_score.device))
            attn_score = F.softmax(attn_scoree, dim=-1)
            updateh = torch.matmul(attn_score, feat_src).permute(0, 2, 1, 3).contiguous()
            new_updateh_shape = updateh.size()[:-2] + (self.out_dim,)
            updateh = updateh.view(new_updateh_shape)
        
            cross_edge_id = self.dst_to_cross_edge_id[(etype, dsttype)]
            if dsttype not in updated_h:
                updated_h[dsttype] = cross_edge_weights[dsttype][cross_edge_id] * updateh
            else:
                updated_h[dsttype] = updated_h[dsttype] + cross_edge_weights[dsttype][cross_edge_id] * updateh

            if srctype == dsttype == 'utterance':
                utterance_to_utterance_matrix = attn_scoree  # batch, heads, utters, utters
            if srctype == 'question' and dsttype == 'utterance':
                utterance_to_question_matrix = attn_scoree # batch, heads, utters, 1
            if dsttype == 'question' and srctype == 'utterance':
                question_to_utterance_matrix = attn_scoree # batch, heads, 1, utters
                
        new_h = {}
        for ntype in self.node_dict:
            n_id = self.node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            trans_out = self.drop(self.a_linears[n_id](updated_h[ntype]))
            trans_out = trans_out * alpha + h[ntype] * (1-alpha)
            new_h[ntype] = self.norms[n_id](trans_out)
        return new_h, utterance_to_utterance_matrix, utterance_to_question_matrix, question_to_utterance_matrix


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, doc_state, nodes_mapping, nodes_len):
        nodes_states = nodes_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        mean_pooled = torch.sum(nodes_states, dim=2) / nodes_len.unsqueeze(2)
        return mean_pooled

class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()
        # self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, doc_state, nodes_mapping, nodes_len):
        """
        :param doc_state:  N x L x d
        :param entity_mapping:  N x E x L
        :param entity_lens:  N x E
        :return: N x E x 2d
        """    # E L 1     1 L D
        entity_states = nodes_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        max_pooled = torch.max(entity_states, dim=2)[0]
        one_vec = torch.ones_like(nodes_len)
        nodes_len = torch.where(nodes_len>0, nodes_len, one_vec)
        mean_pooled = torch.sum(entity_states, dim=2) / nodes_len.unsqueeze(2)
        output = torch.cat([max_pooled, mean_pooled], dim=2)  # N x E x 2d
        return output

class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        '''GatedFusion module'''
        self.fc_z = nn.Linear(4 * hidden_size, hidden_size, bias=True)

    def forward(self, h_state, inputs):
        z = torch.sigmoid(self.fc_z(torch.cat([h_state, inputs, h_state * inputs, h_state - inputs], -1)))
        h_state = (1 - z) * h_state + z * inputs
        return h_state

class ExpandNodesLayer(nn.Module):
    def __init__(self, config):
        super(ExpandNodesLayer, self).__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, nodes_state, nodes_mapping, norm=True):
        """
        :param nodes_state: N x E x H
        :param nodes_mapping: N x E x L
        """
        # E 1 H  *   E, L,1
        expand_nodes_state = torch.sum(nodes_state.unsqueeze(2) * nodes_mapping.unsqueeze(3), dim=1)  
        # N x E x L x H   N x L x H
        if norm:
            expand_nodes_state = self.LayerNorm(expand_nodes_state)
        return expand_nodes_state 


class AttentionPooling2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.GELU(),
                nn.Linear(in_dim, 1),
                )
    
    # @torchsnooper.snoop()
    def forward(self, doc_state, nodes_mapping, nodes_len):
        weights = self.attention(doc_state) # batch, L, 1
        weights = nodes_mapping.unsqueeze(3) * weights.unsqueeze(1)  # N x E x L x 1
        attention_mask = (1.0 - nodes_mapping.to(dtype=next(self.parameters()).dtype)) * -10000.0
        weights = weights + attention_mask.unsqueeze(3)
        weights = torch.nn.functional.softmax(weights, dim=2)
        nodes_states = nodes_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        nodes_states_pooled = torch.sum(nodes_states * weights, dim=2)
        return nodes_states_pooled

class AttentionPooling1(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.GELU(),
                nn.Linear(in_dim, 1),
                )
    
    def forward(
        self,
        hidden_states: torch.Tensor, # B, L, H
        attention_mask: Optional[torch.FloatTensor] = None,  # B,L 
    ) -> Tuple[torch.Tensor]:
        w = self.attention(hidden_states) # B,L,1
        attention_mask = (1.0 - attention_mask.to(dtype=next(self.parameters()).dtype)) * -10000.0
        w = w + attention_mask.unsqueeze(2)
        weights = torch.nn.functional.softmax(w, dim=1)  # B,L,1
        summed_states = torch.sum(hidden_states * weights, dim=1) # B,H
        return summed_states


