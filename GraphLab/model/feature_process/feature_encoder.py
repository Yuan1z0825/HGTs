import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

import GraphLab.register as register
from GraphLab.config import cfg
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)
# Used for the OGB Encoders
full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


class SingleAtomEncoder(torch.nn.Module):
    """
    Only encode the first dimension of atom integer features.
    This feature encodes just the atom type
    Args:
        emb_dim (int): Output embedding dimension
        num_classes: None
    """

    def __init__(self, emb_dim, num_classes=None):
        super(SingleAtomEncoder, self).__init__()

        num_atom_types = full_atom_feature_dims[0]
        self.atom_type_embedding = torch.nn.Embedding(num_atom_types, emb_dim)
        torch.nn.init.xavier_uniform_(self.atom_type_embedding.weight.data)

    def forward(self, batch):
        batch.node_feature = self.atom_type_embedding(batch.node_feature[:, 0])

        return batch


class AtomEncoder(torch.nn.Module):
    """
    The atom Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output embedding dimension
        num_classes: None
    """

    def __init__(self, emb_dim, num_classes=None):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, batch):
        encoded_features = 0
        for i in range(batch.node_feature.shape[1]):
            encoded_features += self.atom_embedding_list[i](
                batch.node_feature[:, i])
        batch.node_feature = encoded_features
        return batch


node_encoder_dict = {
    'SingleAtom': SingleAtomEncoder,
    'Atom': AtomEncoder
}

node_encoder_dict = {**register.node_encoder_dict, **node_encoder_dict}
