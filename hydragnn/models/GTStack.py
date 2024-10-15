##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import global_mean_pool, BatchNorm, Sequential

from .Base import Base, MLPNode
from .GT import GTConv


class GTStack(Base):
    def __init__(self, edge_dim: int, heads: int, pe_dim: int = None, *args, **kwargs):
        self.heads = heads
        self.pe_dim = pe_dim
        self.edge_dim = edge_dim
        super().__init__(*args, **kwargs)

    def _init_conv(self):
        """Here this function overwrites _init_conv() in Base since it has different implementation
        in terms of dimensions due to the multi-head attention"""
        self.node_emb = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        if self.use_pos_emb:
            self.pos_emb = nn.Linear(self.pe_dim, self.hidden_dim, bias=False) #change here for ci tests
        else:
            self.pos_emb = self.register_parameter("pos_emb", None)
        if self.use_edge_attr:
            self.edge_emb = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)
            edge_emb_dim = self.hidden_dim
        else:
            self.edge_emb = self.register_parameter("edge_emb", None)
            edge_emb_dim = None
        self.graph_convs.append(self.get_conv(self.hidden_dim, self.hidden_dim, edge_emb_dim))
        for _ in range(self.num_conv_layers - 2):
            conv = self.get_conv(self.hidden_dim, self.hidden_dim, edge_emb_dim)
            self.graph_convs.append(conv)
        conv = self.get_conv(self.hidden_dim, self.hidden_dim, edge_emb_dim)
        self.graph_convs.append(conv)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the embedding parameters of the model using Xavier uniform initialization.

        Note: The input and the output of the embedding layers does not pass through the activation layer,
              so the variance estimation differs by a factor of two from the default
              kaiming_uniform initialization.
        """
        nn.init.xavier_uniform_(self.node_emb.weight)
        if self.use_pos_emb:
            nn.init.xavier_uniform_(self.pos_emb.weight)
        if self.use_edge_attr:
            nn.init.xavier_uniform_(self.edge_emb.weight)

    def _init_node_conv(self):
        """Here this function overwrites _init_node_conv() in Base since it has different implementation
        in terms of dimensions due to the multi-head attention"""
        # *******convolutional layers for node level predictions*******#
        # two ways to implement node features from here:
        # 1. one graph for all node features
        # 2. one graph for one node features (currently implemented)
        if (
            "node" not in self.config_heads
            or self.config_heads["node"]["type"] != "conv"
        ):
            return
        node_feature_ind = [
            i for i, head_type in enumerate(self.head_type) if head_type == "node"
        ]
        if len(node_feature_ind) == 0:
            return
        if self.use_edge_attr:
            edge_emb_dim = hidden_dim
        else:
            edge_emb_dim = None
        # In this part, each head has same number of convolutional layers, but can have different output dimension
        self.convs_node_hidden.append(
            self.get_conv(self.hidden_dim, self.hidden_dim_node[0], edge_emb_dim)
        )
        for ilayer in range(self.num_conv_layers_node - 1):
            self.convs_node_hidden.append(
                self.get_conv(
                    self.hidden_dim_node[ilayer],
                    self.hidden_dim_node[ilayer + 1],
                    edge_emb_dim,
                )
            )
        for ihead in node_feature_ind:
            self.convs_node_output.append(
                self.get_conv(
                    self.hidden_dim_node[-1], self.head_dims[ihead], edge_emb_dim
                )
            )

    def _multihead(self):
        """Here this function overwrites _multihead() in Base since it has different implementation
        in terms of normalization due to scaled-dot product attention"""
        ############multiple heads/taks################
        # shared dense layers for heads with graph level output
        dim_sharedlayers = 0
        if "graph" in self.config_heads:
            denselayers = []
            dim_sharedlayers = self.config_heads["graph"]["dim_sharedlayers"]
            denselayers.append(nn.Linear(self.hidden_dim, dim_sharedlayers))
            denselayers.append(self.activation_function)
            for ishare in range(self.config_heads["graph"]["num_sharedlayers"] - 1):
                denselayers.append(nn.Linear(dim_sharedlayers, dim_sharedlayers))
                denselayers.append(self.activation_function)
            self.graph_shared = nn.Sequential(*denselayers)

        if "node" in self.config_heads:
            self.num_conv_layers_node = self.config_heads["node"]["num_headlayers"]
            self.hidden_dim_node = self.config_heads["node"]["dim_headlayers"]
            self._init_node_conv()

        inode_feature = 0
        for ihead in range(self.num_heads):
            # mlp for each head output
            if self.head_type[ihead] == "graph":
                num_head_hidden = self.config_heads["graph"]["num_headlayers"]
                dim_head_hidden = self.config_heads["graph"]["dim_headlayers"]
                denselayers = []
                denselayers.append(nn.Linear(dim_sharedlayers, dim_head_hidden[0]))
                denselayers.append(self.activation_function)
                for ilayer in range(num_head_hidden - 1):
                    denselayers.append(
                        nn.Linear(dim_head_hidden[ilayer], dim_head_hidden[ilayer + 1])
                    )
                    denselayers.append(self.activation_function)
                denselayers.append(
                    nn.Linear(
                        dim_head_hidden[-1],
                        self.head_dims[ihead] + self.ilossweights_nll * 1,
                    )
                )
                head_NN = nn.Sequential(*denselayers)
            elif self.head_type[ihead] == "node":
                self.node_NN_type = self.config_heads["node"]["type"]
                head_NN = ModuleList()
                if self.node_NN_type == "mlp" or self.node_NN_type == "mlp_per_node":
                    self.num_mlp = 1 if self.node_NN_type == "mlp" else self.num_nodes
                    assert (
                        self.num_nodes is not None
                    ), "num_nodes must be positive integer for MLP"
                    # """if different graphs in the dataset have different size, one MLP is shared across all nodes """
                    head_NN = MLPNode(
                        self.hidden_dim,
                        self.head_dims[ihead],
                        self.num_mlp,
                        self.hidden_dim_node,
                        self.config_heads["node"]["type"],
                        self.activation_function,
                    )
                elif self.node_NN_type == "conv":
                    for conv in self.convs_node_hidden:
                        head_NN.append(conv)

                    head_NN.append(self.convs_node_output[inode_feature])
                    inode_feature += 1
                else:
                    raise ValueError(
                        "Unknown head NN structure for node features"
                        + self.node_NN_type
                        + "; currently only support 'mlp', 'mlp_per_node' or 'conv' (can be set with config['NeuralNetwork']['Architecture']['output_heads']['node']['type'], e.g., ./examples/ci_multihead.json)"
                    )
            else:
                raise ValueError(
                    "Unknown head type"
                    + self.head_type[ihead]
                    + "; currently only support 'graph' or 'node'"
                )
            self.heads_NN.append(head_NN)

    def forward(self, data):
        """Here this function overwrites forward() in Base since it has different implementation
        in terms of normalization due to scaled-dot product attention"""
        x = data.x.float() #change here
        pos = data.pos

        ### encoder part ####
        if self.use_pos_emb:
            x = self.node_emb(x) + self.pos_emb(data.pe)
        else:
            x = self.node_emb(x)
        conv_args = {"edge_index": data.edge_index.to(torch.long)}
        if self.use_edge_attr:
            assert (
                data.edge_attr is not None
            ), "Data must have edge attributes if use_edge_attributes is set."
            edge_attr = self.edge_emb(data.edge_attr.float().view(-1,self.edge_dim))
            for conv in self.graph_convs:
                conv_args.update({"edge_attr": edge_attr})
                x, edge_attr, pos = conv(x=x, pos=pos, **conv_args)
        else:
            for conv in self.graph_convs:
                x, _, pos = conv(x=x, pos=pos, **conv_args)

        #### multi-head decoder part####
        # shared dense layers for graph level output
        if data.batch is None:
            x_graph = x.mean(dim=0, keepdim=True)
        else:
            x_graph = global_mean_pool(x, data.batch.to(x.device))
        outputs = []
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                x_graph_head = self.graph_shared(x_graph)
                outputs.append(headloc(x_graph_head))
            else:
                if self.node_NN_type == "conv":
                    for conv, batch_norm in zip(headloc[0::2], headloc[1::2]):
                        c, pos = conv(x=x, pos=pos, **conv_args)
                        c = batch_norm(c)
                        x = self.activation_function(c)
                    x_node = x
                else:
                    x_node = headloc(x=x, batch=data.batch)
                outputs.append(x_node)
        return outputs

    def get_conv(self, input_dim, output_dim, edge_dim):
        gt = GTConv(
            node_in_dim=input_dim,
            hidden_dim=output_dim,
            edge_in_dim=edge_dim,
            num_heads=self.heads,
            dropout=self.dropout,
        )
        
        input_args = "x, pos, edge_index"
        conv_args = "x, edge_index"

        if self.use_edge_attr:
            input_args += ", edge_attr"
            conv_args += ", edge_attr"

        return Sequential(
            input_args,
            [
                (gt, conv_args + " -> x, edge_attr"),
                (lambda x, edge_attr, pos: [x, edge_attr, pos], "x, edge_attr, pos -> x, edge_attr, pos"),
            ],
        )

    def __str__(self):
        return "GTStack"

