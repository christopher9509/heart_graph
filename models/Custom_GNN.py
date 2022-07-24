from unittest import result
from sklearn.linear_model import OrthogonalMatchingPursuitCV
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from utils.constants import LayerType
from torch.nn.parameter import Parameter
from utils.utils import *
from torch_scatter import scatter_add
from torch_geometric.data import Data

class MyEnsemble(nn.Module):
    """
    Module that finally derives the output by concatenating the output of the 3-branch models
    (complete)
    """    

    def __init__(self, gat0, gat1, gat2, num_features_per_layer):
        super(MyEnsemble, self).__init__()
        self.gat0 = gat0
        self.gat1 = gat1
        self.gat2 = gat2
        self.out_feat = num_features_per_layer[3]
        
        self.classifier = nn.Linear(self.out_feat*3, 1)
        
    def forward(self, demo_graph, habit_graph, under_graph):
        
        # 3-branch model's output
        head1 = self.gat0(demo_graph)
        head2 = self.gat1(habit_graph)
        head3 = self.gat2(under_graph)
        
        """
        # node-feature matrix extract
        head1 = torch.squeeze(head1[0], 0)
        head2 = torch.squeeze(head2[0], 0)
        head3 = torch.squeeze(head3[0], 0)
        """
        
        # 3-branch concatnate
        x = torch.cat((head1, head2, head3), dim=1)
        result = self.classifier(x)
        
        return result

class HGGNN(torch.nn.Module):

    def __init__(self, num_of_layers, nfeat, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP2, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        GNNLayer = get_layer_type(layer_type)  # fetch one of 3 available implementations
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        GNN_layers1 = []  # collect GNN layers
        GNN_layers2 = []
        GNN_layers3 = []
        
        if GNNLayer == GCNLayer :
            
            layer0 = GNNLayer(
                num_in_features= nfeat,  # consequence of concatenation
                num_out_features= num_features_per_layer[1])
            
            
            layer1 = GNNLayer(
                num_in_features=num_features_per_layer[1],  # consequence of concatenation
                num_out_features=num_features_per_layer[2])
            
            
            layer2 = GNNLayer(
                num_in_features=num_features_per_layer[2],  # consequence of concatenation
                num_out_features=num_features_per_layer[3])
        
        elif GNNLayer == GATLayer :
            layer0 = GNNLayer(
                num_in_features= nfeat * num_heads_per_layer[0],  # consequence of concatenation
                num_out_features= num_features_per_layer[1],
                num_of_heads=num_heads_per_layer[1],
                concat=True,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU(),  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights)
            
            """
            layer1 = GNNLayer(
                num_in_features=num_features_per_layer[1] * num_heads_per_layer[1],  # consequence of concatenation
                num_out_features=num_features_per_layer[2],
                num_of_heads=num_heads_per_layer[2],
                concat=True,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU(),  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights)
            """
            
            layer2 = GNNLayer(
                num_in_features=num_features_per_layer[1] * num_heads_per_layer[1],  # consequence of concatenation
                num_out_features=num_features_per_layer[2],
                num_of_heads=num_heads_per_layer[2],
                concat=False,  # last GAT layer does mean avg, the others do concat
                activation=None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights)
        
        
        GNN_layers1.append(layer0)
        GNN_layers2.append(layer1)
        GNN_layers3.append(layer2)
        
        self.dropout = nn.Dropout(0.5)
        
        self.gnn_net1 = nn.Sequential(*GNN_layers1)
        self.gnn_net2 = nn.Sequential(*GNN_layers2)
        self.gnn_net3 = nn.Sequential(*GNN_layers3)
        
        self.fc = nn.Linear(num_features_per_layer[2], num_features_per_layer[2])

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        
        # 1-layer
        out1 = self.gnn_net1(data)
        out1 = F.relu(out1.x)
        out1 = self.dropout(out1)
        out2 = Data(x=out1, edge_index=data.edge_index)
        
        # 2-layer
        out3 = self.gnn_net2(out2) 
        out4 = F.relu(out3.x)
        out4 = Data(x=out4, edge_index=data.edge_index)
        # 3-layer
        out5 = self.gnn_net3(out4) 

        return out5.x

class GCN__Layer(MessagePassing):
    
    def __init__(self, num_in_features, num_out_features, bias=True, init='xavier'):
        super().__init__()
        self.linear = nn.Linear(num_in_features, num_out_features)
        self.weight = Parameter(torch.Tensor(num_in_features, num_out_features))
#         self.use_bn = bn
#         self.use_atn = atn
#        self.linear = nn.Linear(num_in_features, num_out_features)          

        #nn.init.xavier_uniform_(self.linear.weight)
        #bias = False
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_out_features))
        else:
            self.register_parameter('bias', None)
        
        if init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        else:
            raise NotImplementedError

#        self.reset_parameters()        
        
    def forward(self, x, adj):
        
        
        # 1. Linearly transform node feature matrix  (XΘ)
        # N x emb(out) = N x emb(in) @ emb(in) x emb(out)
        x = torch.matmul(x, self.weight)
        
        # 2. Add self-loops to the adjacency matrix   (A' = A + I)
        adj = add_self_loops(adj, x.size(0))    # 2 x (E+N)
        
        # 3. Compute normalization  ( D^(-0.5) A D^(-0.5) )
        edge_weight = torch.ones((adj.size(1),),
                                 dtype=x.dtype,
                                 device=adj.device)  # [E+N]
        row, col = adj  # [E+N], [E+N]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))  # [N]
        deg_inv_sqrt = deg.pow(-0.5)  # [N]
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # [N]  # same to use masked_fill_
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # [E+N]
        
        # 4. Start propagating messages
        return self.propagate(adj, x=x, norm=norm)  # 2 x (E+N), N x emb(out), [E+N]

    def message(self, x_j, norm):  # 4.1 Normalize node features.
        # x_j: after linear x and expand edge  (N+E) x emb(out) = N x emb(in) @ emb(in) x emb(out) (+) E x emb(out)
        return norm.view(-1, 1) * x_j  # (N+E) x emb(out)  
        # return: each row is norm(embedding) vector for each edge_index pair

    def update(self, aggr_out):  # 4.2 Return node embeddings)
        # for Node 0: Based on the directed graph, Node 0 gets message from three edges and one self_loop
        # for Node 1, 2, 3: since they do not get any message from others, so only self_loop
        if self.bias is not None:
            return aggr_out + self.bias
        else:
            return aggr_out

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0) 
            
#    def reset_parameters(self):
#        glorot(self.weight)
#        zeros(self.bias)

class GCNLayer(torch.nn.Module):
    
    def __init__(self, num_in_features, num_out_features):
        super().__init__()
        self.conv = GCNConv(num_in_features, num_out_features)

    def forward(self, data):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x_, edge_index = data.x, data.edge_index
        
        x_ = self.conv(x_, edge_index)
        out_data = Data(x=x_, edge_index=edge_index)
        return out_data



class GATLayer(torch.nn.Module):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3

    """

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0, add_skip_connection=True, bias=True, log_attention_weights=False):

        # Delegate initialization to the base class
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation, dropout_prob,
                      add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features, edge_index = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        ## attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'

    if layer_type == LayerType.IMP1:
        return GATLayer # Graph Attention Network
    elif layer_type == LayerType.IMP2:
        return GCNLayer # Graph Convolution Network
    elif layer_type == LayerType.IMP3:
        return HGGAT # Graph Custom Netwrok
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')