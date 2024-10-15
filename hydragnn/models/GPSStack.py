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
from torch_geometric.nn import global_mean_pool, BatchNorm, Sequential, GINConv, GINEConv, PNAConv
from torch_geometric.nn.attention import PerformerAttention
from typing import Optional
from .Base import Base, MLPNode
from .GPS import GPSConv

class GPSStack(Base):
    def __init__(self, deg: list, edge_dim: int, heads: int, attn_type: str, pe_dim: int = None, *args, **kwargs):
        self.heads = heads
        self.pe_dim = pe_dim
        self.edge_dim = edge_dim
        self.attn_type = attn_type
        self.aggregators = ["mean", "min", "max", "std"]
        self.scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]
        self.deg = torch.Tensor(deg)
        super().__init__(*args, **kwargs)

    def _init_conv(self):
        """Here this function overwrites _init_conv() in Base since it has different implementation
        in terms of dimensions due to the multi-head attention"""
        self.node_emb = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        if self.use_pos_emb:
            self.pos_emb = nn.Linear(self.pe_dim, self.hidden_dim, bias=False) #change here for ci tests
            self.node_lin = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False)
        else:
            self.pos_emb = self.register_parameter("pos_emb", None)
            self.node_lin = self.register_parameter("node_lin", None)
        if self.use_edge_attr:
            self.edge_emb = nn.Linear(self.edge_dim, self.hidden_dim, bias=False)
        else:
            self.edge_emb = self.register_parameter("edge_emb", None)
        self.graph_convs.append(self.get_conv(self.hidden_dim, self.hidden_dim))
        for _ in range(self.num_conv_layers - 2):
            conv = self.get_conv(self.hidden_dim, self.hidden_dim)
            self.graph_convs.append(conv)
        conv = self.get_conv(self.hidden_dim, self.hidden_dim)
        self.graph_convs.append(conv)
        self.redraw_projection = RedrawProjection(
            self.graph_convs,
            redraw_interval=1000 if self.attn_type == 'performer' else None)
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
            nn.init.xavier_uniform_(self.node_lin.weight)
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

        # In this part, each head has same number of convolutional layers, but can have different output dimension
        self.convs_node_hidden.append(
            self.get_conv(self.hidden_dim, self.hidden_dim_node[0])
        )
        for ilayer in range(self.num_conv_layers_node - 1):
            self.convs_node_hidden.append(
                self.get_conv(
                    self.hidden_dim_node[ilayer],
                    self.hidden_dim_node[ilayer + 1],
                )
            )
        for ihead in node_feature_ind:
            self.convs_node_output.append(
                self.get_conv(
                    self.hidden_dim_node[-1], self.head_dims[ihead]
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
        x = data.x.float() 
        pos = data.pos
        batch = data.batch
        
        ### encoder part ####
        if self.use_pos_emb:
            x = torch.cat((self.node_emb(x), self.pos_emb(data.pe)), 1) 
            x = self.node_lin(x)
        else:
            x = self.node_emb(x)
         
        edge_index = data.edge_index.to(torch.long)
        if self.use_edge_attr:
            assert (
                data.edge_attr is not None
            ), "Data must have edge attributes if use_edge_attributes is set."
            edge_attr = self.edge_emb(data.edge_attr.float().view(-1,self.edge_dim)) #change here
            for conv in self.graph_convs:
                x, pos = conv(x=x, pos=pos, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        else:
            for conv in self.graph_convs:
                x, pos = conv(x=x, pos=pos, edge_index=edge_index, batch=batch)

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
                        c, pos = conv(x=x, pos=pos, edge_index=edge_index, batch=batch)
                        c = batch_norm(c)
                        x = self.activation_function(c)
                    x_node = x
                else:
                    x_node = headloc(x=x, batch=data.batch)
                outputs.append(x_node)
        return outputs

    def get_conv(self, input_dim, output_dim):
        mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )

        # if self.use_edge_attr:
        #     conv = GINEConv(mlp,eps=100.0,train_eps=True)
        # else:
        conv = GINConv(mlp,eps=100.0,train_eps=True)
        # conv = PNAConv(
        #     in_channels=input_dim,
        #     out_channels=output_dim,
        #     aggregators=self.aggregators,
        #     scalers=self.scalers,
        #     deg=self.deg,
        #     edge_dim=input_dim,#self.edge_dim,
        #     pre_layers=1,
        #     post_layers=1,
        #     divide_input=False,
        # )

        gps = GPSConv(
            channels=input_dim,
            conv=conv,
            heads=self.heads,
            dropout=self.dropout,
            attn_type=self.attn_type,
        )
        
        input_args = "x, pos, edge_index, batch"
        conv_args = "x, edge_index, batch"

        if self.use_edge_attr:
            input_args += ", edge_attr"
            conv_args += ", edge_attr"

        return Sequential(
            input_args,
            [
                (gps, conv_args + " -> x"),
                (lambda x, pos: [x, pos], "x, pos -> x, pos"),
            ],
        )

    def __str__(self):
        return "GPSStack"

class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1