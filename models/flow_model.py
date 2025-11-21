"""Neural network architecture for the flow model."""
import torch
from torch import nn
from types import SimpleNamespace
from typing import Callable, Literal, Optional, Tuple, Union
from models.node_embedder import NodeEmbedder,NodeEmbedder_v2
from models.edge_embedder import EdgeEmbedder
from models import ipa_pytorch,GNN
from data import utils as du
from chroma.layers import basic
from chroma.layers.structure import protein_graph
from chroma.layers.structure.backbone import FrameBuilder
from data.interpolant import save_pdb_chain
class BackboneEncoderGNN(nn.Module):
    """Graph Neural Network for processing protein structure into graph embeddings.

    Args:
        See documention of `structure.protein_graph.ProteinFeatureGraph`,
        and `graph.GraphNN` for more details.

        dim_nodes (int): Hidden dimension of node tensors.
        dim_edges (int): Hidden dimension of edge tensors.
        num_neighbors (int): Number of neighbors per nodes.
        node_features (tuple): List of node feature specifications. Features
            can be given as strings or as dictionaries.
        edge_features (tuple): List of edge feature specifications. Features
            can be given as strings or as dictionaries.
        num_layers (int): Number of layers.
        node_mlp_layers (int): Number of hidden layers for node update
            function.
        node_mlp_dim (int, optional): Dimension of hidden layers for node update
            function, defaults to match output dimension.
        edge_update (bool): Whether to include an edge update step.
        edge_mlp_layers (int): Number of hidden layers for edge update
            function.
        edge_mlp_dim (int, optional): Dimension of hidden layers for edge update
            function, defaults to match output dimension.
        skip_connect_input (bool): Whether to include skip connections between
            layers.
        mlp_activation (str): MLP nonlinearity function, `relu` or `softplus`
            accepted.
        dropout (float): Dropout fraction.
        graph_distance_atom_type (int): Atom type for computing residue-residue
            distances for graph construction. Negative values will specify
            centroid across atom types. Default is `-1` (centroid).
        graph_cutoff (float, optional): Cutoff distance for graph construction:
            mask any edges further than this cutoff. Default is `None`.
        graph_mask_interfaces (bool): Restrict connections only to within
            chains, excluding-between chain interactions. Default is `False`.
        graph_criterion (str): Method used for building graph from distances.
            Currently supported methods are `{knn, random_log, random_linear}`.
            Default is `knn`.
        graph_random_min_local (int): Minimum number of neighbors in GNN that
            come from local neighborhood, before random neighbors are chosen.
        checkpoint_gradients (bool): Switch to implement gradient checkpointing
            during training.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
        C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
        node_h_aux (torch.LongTensor, optional): Auxiliary node features with
            shape `(num_batch, num_residues, dim_nodes)`.
        edge_h_aux (torch.LongTensor, optional): Auxiliary edge features with
            shape `(num_batch, num_residues, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor, optional): Input edge indices for neighbors
            with shape `(num_batch, num_residues, num_neighbors)`.
        mask_ij (torch.Tensor, optional): Input edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.

    Outputs:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_residues, dim_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_residues, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_residues, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_residues)`.
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.
    """

    def __init__(
        self,
        dim_nodes: int = 256,
        dim_edges: int = 128,
        num_neighbors: int = 30,
        node_features: tuple = (("internal_coords", {"log_lengths": True}),),
        edge_features: tuple = (
            "distances_2mer",
            "orientations_2mer",
            "distances_chain",
        ),
        num_layers: int = 3,
        node_mlp_layers: int = 1,
        node_mlp_dim: Optional[int] = None,
        edge_update: bool = True,
        edge_mlp_layers: int = 1,
        edge_mlp_dim: Optional[int] = None,
        skip_connect_input: bool = False,
        mlp_activation: str = "softplus",
        dropout: float = 0.1,
        graph_distance_atom_type: int = -1,
        graph_cutoff: Optional[float] = None,
        graph_mask_interfaces: bool = False,
        graph_criterion: str = "knn",
        graph_random_min_local: int = 20,
        checkpoint_gradients: bool = False,
        **kwargs
    ) -> None:
        """Initialize BackboneEncoderGNN."""
        super(BackboneEncoderGNN, self).__init__()

        # Save configuration in kwargs
        self.kwargs = locals()
        self.kwargs.pop("self")
        for key in list(self.kwargs.keys()):
            if key.startswith("__") and key.endswith("__"):
                self.kwargs.pop(key)
        args = SimpleNamespace(**self.kwargs)

        # Important global options
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.checkpoint_gradients = checkpoint_gradients

        graph_kwargs = {
            "distance_atom_type": args.graph_distance_atom_type,
            "cutoff": args.graph_cutoff,
            "mask_interfaces": args.graph_mask_interfaces,
            "criterion": args.graph_criterion,
            "random_min_local": args.graph_random_min_local,
        }

        self.feature_graph = protein_graph.ProteinFeatureGraph(
            dim_nodes=args.dim_nodes,
            dim_edges=args.dim_edges,
            num_neighbors=args.num_neighbors,
            graph_kwargs=graph_kwargs,
            node_features=args.node_features,
            edge_features=args.edge_features,
        )




    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        node_h_aux: Optional[torch.Tensor] = None,
        edge_h_aux: Optional[torch.Tensor] = None,
        edge_idx: Optional[torch.Tensor] = None,
        mask_ij: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor
    ]:
        """Encode XC backbone structure into node and edge features."""
        num_batch, num_residues = C.shape

        # 生成所有可能的残基对的索引
        i_idx, j_idx = torch.meshgrid(torch.arange(num_residues), torch.arange(num_residues), indexing='ij')
        # 扩展 edge_idx 以匹配批次大小
        edge_idx = j_idx.unsqueeze(0).expand(num_batch, num_residues, num_residues).to(X.device)

        # Hack to enable checkpointing
        if self.checkpoint_gradients and (not X.requires_grad):
            X.requires_grad = True

        node_h, edge_h, edge_idx, mask_i, mask_ij = self.feature_graph( X, C, edge_idx, mask_ij)

        if node_h_aux is not None:
            node_h = node_h + mask_i.unsqueeze(-1) * node_h_aux
        if edge_h_aux is not None:
            edge_h = edge_h + mask_ij.unsqueeze(-1) * edge_h_aux

        return node_h, edge_h, edge_idx, mask_i, mask_ij


class FlowModel_binder(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel_binder, self).__init__()

        # self._set_static_graph()

        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_embedder = NodeEmbedder_v2(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

        # Time features
        dim_nodes = self._ipa_conf.c_s


        # Feature trunk
        self.feature_graph = BackboneEncoderGNN(dim_nodes=dim_nodes)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        self.gnn=GNN.Graph(node_features=self._model_conf.node_features.c_s,
                           edge_features=self._model_conf.edge_features.c_p,
                           hidden_dim=self._ipa_conf.c_hidden,
                           precision=self._model_conf.precision,
                           ipa=self._ipa_conf,
                           num_encoder_layers=3)  #


        self.frame_builder = FrameBuilder()
    def _feature_trunk(self, X, chain_idx):
        """
        Applies the feature trunk to the input features.
        Args:
            X (torch.Tensor): Input features.
            chain_idx (torch.Tensor): Chain indices.
        Returns:
            node_h (torch.Tensor): Output node features.
            edge_h (torch.Tensor): Output edge features.
            edge_idx (torch.Tensor): Edge indices.
            mask_i (torch.Tensor): Mask for node features.
            mask_ij (torch.Tensor): Mask for edge features.

        """
        # 假设 X 是输入的原子坐标张量，形状为 [B, N, 4, 3]
        num_batch = X.shape[0]
        num_residues = X.shape[1]
        node_h, edge_h, edge_idx, mask_i, mask_ij=self.feature_graph(X, chain_idx)

        return node_h, edge_h, edge_idx, mask_i, mask_ij


    def _preprocess(self,input_feats,is_training):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        continuous_t = input_feats['t']  # .half()

        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']

        chain_idx = input_feats['chain_idx']

        ######if 'bbatoms_t'  not in input_feats:
        X_t = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        # else:
        #     X_t = input_feats['bbatoms_t']  # noised backbone


        ##########save######

        #save_pdb_chain(X_t[0].clone().reshape(-1,3).cpu().numpy(), chain_idx[0].clone().cpu().numpy(), 'X_t.pdb')

        node_h, edge_h, edge_idx, mask_i, mask_ij = self._feature_trunk(X_t, chain_idx )

        num_res = chain_idx.shape[1]
        pos = torch.arange(num_res).long().to(chain_idx.device).unsqueeze(0).repeat(chain_idx.shape[0], 1)
        # pos=input_feats['res_idx'].long()
        # Initialize node and edge embeddings
        if 'fixed_mask'  not in input_feats:
            fixed_mask=None
        else:
            fixed_mask=input_feats['fixed_mask']
        init_node_embed = self.node_embedder(continuous_t, node_mask,is_training,**input_feats)
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask)

        init_node_embed = init_node_embed + node_h
        init_edge_embed = init_edge_embed + edge_h

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t, )

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

        return curr_rigids,node_embed,edge_embed,pos,node_mask,edge_mask,chain_idx

    def forward(self, input_feats,recycle=1,is_training=True,runfor_fbb=False):

        #preprocess feature
        curr_rigids, node_embed, edge_embed, pos, node_mask,edge_mask,chain_idx=self._preprocess(input_feats,is_training)


        #main trunk
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed,e = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                pos,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]


            if input_feats['fixed_mask'] is not None:
                update_mask=node_mask[..., None]*(1-input_feats['fixed_mask'][...,None])   # mark the pos which needed to be update
            else:
                update_mask=node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * update_mask)
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, update_mask)

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]


        # gnn this is older version  where i input ,node_h,edge_, this seems bad
        # curr_rigids = self.gnn( chain_idx, curr_rigids,node_init_embed,edge_init_embed,pos,node_mask,mode,**input_feats)

        # # try update feature
        # curr_rigids = self.gnn( chain_idx, curr_rigids,node_embed,edge_embed,pos,node_mask,mode,**input_feats)

        # try mixed feature
        # new features


        node_embed=node_embed#+node_init_embed
        edge_embed=edge_embed#+edge_init_embed



        # Final rigids

        if not runfor_fbb:
            # print(' run in binder mode, and use forward_fix_tpology_forbinder')
            curr_rigids, b_f, chi, S = self.gnn.forward_fix_tpology_forbinder(chain_idx, curr_rigids, node_embed, edge_embed, pos, node_mask,
                                                      recycle, self._feature_trunk, self.frame_builder,
                                                      self.rigids_nm_to_ang, **input_feats)

            # Final rigids
            curr_rigids = self.rigids_nm_to_ang(curr_rigids)
            pred_trans = curr_rigids.get_trans()
            pred_rotmats = curr_rigids.get_rots().get_rot_mats()

            # X_t=self.frame_builder(pred_rotmats, pred_trans, chain_idx).float()
            # save_pdb_chain(X_t[0].clone().reshape(-1,3).cpu().numpy(), chain_idx[0].clone().cpu().numpy(), 'X_out.pdb')

            ##BF
            if b_f is None:
                return {
                    'pred_trans': pred_trans,
                    'pred_rotmats': pred_rotmats, }
            else:
                b_f = b_f * (1 - input_feats['fixed_mask'][..., None]) + (
                            input_feats['atoms14_b_factors'][..., :4] / 20) * input_feats['fixed_mask'][..., None]
                chi = chi * (1 - input_feats['fixed_mask'][..., None]) + input_feats['chi'] * input_feats['fixed_mask'][
                    ..., None]

                return {
                    'pred_trans': pred_trans,
                    'pred_rotmats': pred_rotmats,
                    'pred_bf': b_f,
                    'pred_chi': chi,
                    'logits': S,
                    'SEQ': torch.argmax(S, dim=-1) * (1 - input_feats['fixed_mask']) + input_feats['aatype'] *
                           input_feats['fixed_mask']
                }

        else:
            # for fbb design , so output more information
            curr_rigids, b_f, chi, S, h_V = self.gnn(chain_idx, curr_rigids, node_embed, edge_embed, pos,
                                                          node_mask,
                                                          recycle, self._feature_trunk, self.frame_builder,
                                                          self.rigids_nm_to_ang, **input_feats)

            #curr_rigids nm
            curr_rigids = self.rigids_nm_to_ang(curr_rigids)
            pred_trans = curr_rigids.get_trans()
            pred_rotmats = curr_rigids.get_rots().get_rot_mats()

            return pred_trans,pred_rotmats, b_f





class FlowModel(nn.Module):

    def __init__(self, model_conf,mode='motif'):

        print('current design mode is ',mode)

        super(FlowModel, self).__init__()
        self._model_conf = model_conf

        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_embedder = NodeEmbedder(model_conf.node_features,mode)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

        self.mode=mode
        # Time features
        dim_nodes = self._ipa_conf.c_s
        # if mode=='base':
        #     #basemodel是比较老的那一版，里面有这种无用的权重
        #     self.time_features = basic.FourierFeaturization(
        #         d_input=1, d_model=dim_nodes, trainable=False, scale=16.0
        #     )
        # else:
        #     # in mode base , no frame builder register
        #     self.frame_builder = FrameBuilder()
        self.frame_builder = FrameBuilder()
        # Feature trunk
        self.feature_graph = BackboneEncoderGNN(dim_nodes=dim_nodes)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.1,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )
        #self._model_conf.precision=32
        self.gnn=GNN.Graph(node_features=self._model_conf.node_features.c_s,
                           edge_features=self._model_conf.edge_features.c_p,
                           hidden_dim=self._ipa_conf.c_hidden,
                            num_encoder_layers=3,
                           precision=self._model_conf.precision,
                           ipa=self._ipa_conf,
                           CSB=False,
                           mode=self.mode)  #


    def _feature_trunk(self, X, chain_idx):
        """
        Applies the feature trunk to the input features.
        Args:
            X (torch.Tensor): Input features.
            chain_idx (torch.Tensor): Chain indices.
        Returns:
            node_h (torch.Tensor): Output node features.
            edge_h (torch.Tensor): Output edge features.
            edge_idx (torch.Tensor): Edge indices.
            mask_i (torch.Tensor): Mask for node features.
            mask_ij (torch.Tensor): Mask for edge features.

        """
        # 假设 X 是输入的原子坐标张量，形状为 [B, N, 4, 3]
        num_batch = X.shape[0]
        num_residues = X.shape[1]
        node_h, edge_h, edge_idx, mask_i, mask_ij=self.feature_graph(X, chain_idx)

        return node_h, edge_h, edge_idx, mask_i, mask_ij

    def _preprocess(self,input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        continuous_t = input_feats['t']  # .half()

        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']

        chain_idx = input_feats['chain_idx']

        if 'bbatoms_t'  not in input_feats:
            X_t = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        else:
            X_t = input_feats['bbatoms_t']  # noised backbone

        node_h, edge_h, edge_idx, mask_i, mask_ij = self._feature_trunk(X_t, chain_idx, )

        num_res = chain_idx.shape[1]
        pos = torch.arange(num_res).long().to(chain_idx.device).unsqueeze(0).repeat(chain_idx.shape[0], 1)
        # pos=input_feats['res_idx'].long()
        # Initialize node and edge embeddings
        if 'fixed_mask'  not in input_feats:
            fixed_mask=None
        else:
            fixed_mask=input_feats['fixed_mask']

        if self.mode=='base_ss':
            init_node_embed = self.node_embedder(continuous_t, node_mask, fixed_mask,input_feats['ss'])
        else:
            init_node_embed = self.node_embedder(continuous_t, node_mask,fixed_mask)
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask)

        init_node_embed = init_node_embed + node_h  # .half()
        init_edge_embed = init_edge_embed + edge_h

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t, )

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

        return curr_rigids,node_embed,edge_embed,pos,node_mask,edge_mask,chain_idx

    def forward(self, input_feats,recycle=1):

        #preprocess feature
        curr_rigids, node_embed, edge_embed, pos, node_mask,edge_mask,chain_idx=self._preprocess(input_feats)

        # node_init_embed=node_embed
        # edge_init_embed = edge_embed

        #main trunk
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed,e = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                pos,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]


            if input_feats['fixed_mask'] is not None:
                update_mask=node_mask[..., None]*(1-input_feats['fixed_mask'][...,None])   # mark the pos which needed to be update
            else:
                update_mask=node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * update_mask)
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, update_mask)

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]


        # gnn this is older version  where i input ,node_h,edge_, this seems bad
        # curr_rigids = self.gnn( chain_idx, curr_rigids,node_init_embed,edge_init_embed,pos,node_mask,mode,**input_feats)

        # # try update feature
        # curr_rigids = self.gnn( chain_idx, curr_rigids,node_embed,edge_embed,pos,node_mask,mode,**input_feats)

        # try mixed feature
        # new features
        from chroma.data.protein import Protein

        curr_rigids_cp = self.rigids_nm_to_ang(curr_rigids)
        trans_t_2 = curr_rigids_cp.get_trans()
        rotmats_t_2 = curr_rigids_cp.get_rots().get_rot_mats()
        atoms4 = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
       # p = Protein.from_XCS(atoms4.detach().cpu(), chain_idx.detach().cpu(), chain_idx.detach().cpu(), )
        # p.to_PDB('/home/junyu/project/Proflow/unmidout/' + str('test_') +str(t_value)+ '.pdb')


        node_embed=node_embed#+node_init_embed  #
        edge_embed=edge_embed#+edge_init_embed  #
        if self.mode=='base'or self.mode=='base_ss':  #
            '''
            actually we had change base model also to forward_fix_tpology
            '''
            curr_rigids = self.gnn.forward_fix_tpology(chain_idx, curr_rigids, node_embed, edge_embed, pos, node_mask,
                                                       recycle, **input_feats)
            # curr_rigids = self.gnn.forwardSS(chain_idx, curr_rigids, node_embed, edge_embed, pos, node_mask, recycle,
            #                                  self._preprocess, self.rigids_nm_to_ang, **input_feats)

        else: ## motif mode
            if 'gnn_update' in  self._model_conf and self._model_conf.gnn_update=='fixtopo':  # gnn_update

                # curr_rigids = self.gnn( chain_idx, curr_rigids,node_embed,edge_embed,pos,node_mask,mode,**input_feats)

                curr_rigids = self.gnn.forward_fix_tpology(chain_idx, curr_rigids, node_embed, edge_embed, pos,
                                                           node_mask,
                                                           recycle, **input_feats)
            else:
                print(' use forwardSS')
                curr_rigids = self.gnn.forwardSS( chain_idx, curr_rigids,node_embed,edge_embed,pos,node_mask,recycle,self._preprocess,self.rigids_nm_to_ang,**input_feats)

        # Final rigids
        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

      
        atoms4 = self.frame_builder(pred_rotmats.float(), pred_trans, chain_idx)
        p = Protein.from_XCS(atoms4.detach().cpu(), chain_idx.detach().cpu(), chain_idx.detach().cpu(), )
        # p.to_PDB('/home/junyu/project/Proflow/unout/' + str('test_') +str(t_value) +'.pdb')


        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
        }



class FlowModel_binder_sidechain(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel_binder_sidechain, self).__init__()

        self._model_conf=model_conf
        self._ipa_conf = model_conf.ipa

        self.node_embedder = NodeEmbedder_v2(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)
        # Feature trunk
        self.feature_graph = BackboneEncoderGNN()
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)

        self.gnn=GNN.Sidechain(node_features=self._model_conf.node_features.c_s,
                               edge_features=self._model_conf.edge_features.c_p,
                               hidden_dim=self._ipa_conf.c_hidden,
                               precision=self._model_conf.precision,
                               ipa=self._ipa_conf)  #


        self.frame_builder = FrameBuilder()
    def _feature_trunk(self, X, chain_idx):
        """
        Applies the feature trunk to the input features.
        Args:
            X (torch.Tensor): Input features.
            chain_idx (torch.Tensor): Chain indices.
        Returns:
            node_h (torch.Tensor): Output node features.
            edge_h (torch.Tensor): Output edge features.
            edge_idx (torch.Tensor): Edge indices.
            mask_i (torch.Tensor): Mask for node features.
            mask_ij (torch.Tensor): Mask for edge features.

        """
        # 假设 X 是输入的原子坐标张量，形状为 [B, N, 4, 3]
        num_batch = X.shape[0]
        num_residues = X.shape[1]
        node_h, edge_h, edge_idx, mask_i, mask_ij=self.feature_graph(X, chain_idx)

        return node_h, edge_h, edge_idx, mask_i, mask_ij


    def _preprocess_fbb(self,input_feats,is_training):


        node_mask = input_feats['res_mask']
        continuous_t = input_feats['t']

        #use output bf
        init_node_embed = self.node_embedder.forward_forfbb(continuous_t, node_mask,is_training,**input_feats)
        chain_idx = input_feats['chain_idx']
        return init_node_embed,node_mask,chain_idx


    def _preprocess(self,input_feats,is_training):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        continuous_t = input_feats['t']  # .half()

        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']

        chain_idx = input_feats['chain_idx']

        ######if 'bbatoms_t'  not in input_feats:
        X_t = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        # else:
        #     X_t = input_feats['bbatoms_t']  # noised backbone


        ##########save######

        #save_pdb_chain(X_t[0].clone().reshape(-1,3).cpu().numpy(), chain_idx[0].clone().cpu().numpy(), 'X_t.pdb')

        node_h, edge_h, edge_idx, mask_i, mask_ij = self._feature_trunk(X_t, chain_idx )

        num_res = chain_idx.shape[1]
        pos = torch.arange(num_res).long().to(chain_idx.device).unsqueeze(0).repeat(chain_idx.shape[0], 1)
        # pos=input_feats['res_idx'].long()
        # Initialize node and edge embeddings
        if 'fixed_mask'  not in input_feats:
            fixed_mask=None
        else:
            fixed_mask=input_feats['fixed_mask']
        init_node_embed = self.node_embedder.forward_forfbb(continuous_t, node_mask, is_training, **input_feats)
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask)

        init_node_embed = init_node_embed + node_h
        init_edge_embed = init_edge_embed + edge_h

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t, )

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

        return curr_rigids,node_embed,edge_embed,pos,node_mask,edge_mask,chain_idx

    def forward(self, input_feats,
                is_training=True,):

        #### ss sidechain information encoder
        #node_embed_forfbb,node_mask,chain_idx=self._preprocess(input_feats,is_training)
        curr_rigids, node_embed, edge_embed, pos, node_mask,edge_mask,chain_idx=self._preprocess(input_feats,is_training)


        curr_rigids, chi, S = self.gnn(chain_idx, curr_rigids, node_embed, edge_embed, pos, node_mask,
                                            self._feature_trunk, self.frame_builder,
                                            self.rigids_nm_to_ang, **input_feats)

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()


        chi = chi * (1 - input_feats['fixed_mask'][..., None]) + input_feats['chi'] * input_feats['fixed_mask'][
                            ..., None]

        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'pred_chi': chi,
            'logits': S,
            'SEQ': torch.argmax(S, dim=-1) * (1 - input_feats['fixed_mask']) + input_feats['aatype'] *
                   input_feats['fixed_mask']
        }


class FlowModel_seqdesign(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel_seqdesign, self).__init__()

        self._model_conf=model_conf
        self._ipa_conf = model_conf.ipa

        self.node_embedder = NodeEmbedder(model_conf.node_features,mode='fbb')
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)
        # Feature trunk
        self.feature_graph = BackboneEncoderGNN()
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)

        self.gnn=GNN.Sidechain(node_features=self._model_conf.node_features.c_s,
                               edge_features=self._model_conf.edge_features.c_p,
                               hidden_dim=self._ipa_conf.c_hidden,
                               precision=self._model_conf.precision,
                               ipa=self._ipa_conf)  #


        self.frame_builder = FrameBuilder()
    def _feature_trunk(self, X, chain_idx):
        """
        Applies the feature trunk to the input features.
        Args:
            X (torch.Tensor): Input features.
            chain_idx (torch.Tensor): Chain indices.
        Returns:
            node_h (torch.Tensor): Output node features.
            edge_h (torch.Tensor): Output edge features.
            edge_idx (torch.Tensor): Edge indices.
            mask_i (torch.Tensor): Mask for node features.
            mask_ij (torch.Tensor): Mask for edge features.

        """
        # 假设 X 是输入的原子坐标张量，形状为 [B, N, 4, 3]
        num_batch = X.shape[0]
        num_residues = X.shape[1]
        node_h, edge_h, edge_idx, mask_i, mask_ij=self.feature_graph(X, chain_idx)

        return node_h, edge_h, edge_idx, mask_i, mask_ij


    def _preprocess_fbb(self,input_feats,is_training):


        node_mask = input_feats['res_mask']
        continuous_t = input_feats['t']

        #use output bf
        init_node_embed = self.node_embedder.forward_forfbb(continuous_t, node_mask,is_training,**input_feats)
        chain_idx = input_feats['chain_idx']
        return init_node_embed,node_mask,chain_idx


    def _preprocess(self,input_feats,is_training):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        continuous_t = input_feats['t']  # .half()

        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']

        chain_idx = input_feats['chain_idx']

        ######if 'bbatoms_t'  not in input_feats:
        X_t = input_feats['bbatoms_t']
        # else:
        #     X_t = input_feats['bbatoms_t']  # noised backbone


        ##########save######

        #save_pdb_chain(X_t[0].clone().reshape(-1,3).cpu().numpy(), chain_idx[0].clone().cpu().numpy(), 'X_t.pdb')

        node_h, edge_h, edge_idx, mask_i, mask_ij = self._feature_trunk(X_t, chain_idx )

        num_res = chain_idx.shape[1]
        pos = torch.arange(num_res).long().to(chain_idx.device).unsqueeze(0).repeat(chain_idx.shape[0], 1)
        # pos=input_feats['res_idx'].long()
        # Initialize node and edge embeddings
        if 'fixed_mask'  not in input_feats:
            fixed_mask=None
        else:
            fixed_mask=input_feats['fixed_mask']
        init_node_embed = self.node_embedder(continuous_t, node_mask, fixed_mask)
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_embedder(
            init_node_embed, trans_t, trans_sc, edge_mask)

        init_node_embed = init_node_embed + node_h
        init_edge_embed = init_edge_embed + edge_h

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t, )

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

        return curr_rigids,node_embed,edge_embed,pos,node_mask,edge_mask,chain_idx

    def forward(self, input_feats,
                is_training=True,):

        #### ss sidechain information encoder
        #node_embed_forfbb,node_mask,chain_idx=self._preprocess(input_feats,is_training)
        curr_rigids, node_embed, edge_embed, pos, node_mask,edge_mask,chain_idx=self._preprocess(input_feats,is_training)


        curr_rigids, chi, S = self.gnn.forward_fixed(chain_idx, curr_rigids, node_embed, edge_embed, pos, node_mask,
                                            self._feature_trunk, self.frame_builder,
                                            self.rigids_nm_to_ang, **input_feats)



        chi = chi * input_feats['res_mask'][ ..., None]

        return {

            'pred_chi': chi,
            'logits': S,
            'SEQ': torch.argmax(S, dim=-1)
        }