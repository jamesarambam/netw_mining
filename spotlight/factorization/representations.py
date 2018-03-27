"""
Classes defining user and item latent representations in
factorization models.
"""

import torch.nn as nn
from torch.autograd import Variable
from spotlight.layers import ScaledEmbedding, ZeroEmbedding
import torch


class BilinearNet(nn.Module):
    """
    Bilinear factorization representation.
    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.
    Parameters
    ----------
    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 user_embedding_layer=None, item_embedding_layer=None, sparse=False):

        super(BilinearNet, self).__init__()

        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   sparse=sparse)

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.
        Parameters
        ----------
        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.
        Returns
        -------
        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        return dot + user_bias + item_bias

class RankingNet(nn.Module):
    """
    Bilinear factorization representation.

    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.

    """

    def __init__(self, num_users, num_items, embedding_dim=32,
                 user_embedding_layer=None, item_embedding_layer=None, sparse=False):

        super(RankingNet, self).__init__()

        self.embedding_dim = embedding_dim

        self.inputDim = embedding_dim * 3
        self.hiddenDim = self.inputDim * 2
        self.hiddenDim2 = embedding_dim * 3
        self.l2Output = 1


        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                                   sparse=sparse)

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   sparse=sparse)

        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

        # self.linear1 = nn.Linear(self.inputDim, self.hiddenDim)
        # self.bn1 = nn.BatchNorm1d(self.hiddenDim)
        # self.linear2 = nn.Linear(self.hiddenDim, self.l2Output)
        # self.dropout = nn.Dropout(p = 0.8)
        # self.output = nn.Sigmoid()

        self.linear1 = nn.Linear(self.inputDim, self.hiddenDim)
        self.bn1 = nn.BatchNorm1d(self.hiddenDim)
        self.linear2 = nn.Linear(self.hiddenDim, self.hiddenDim2)
        self.bn2 = nn.BatchNorm1d(self.hiddenDim2)
        self.linear3 = nn.Linear(self.hiddenDim2, self.l2Output)
        self.dropout = nn.Dropout(p = 0.8)
        self.output = nn.Sigmoid()


    def forward(self, x):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        pos_item_ids: tensor
            Tensor of item indices.
        neg_item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """
        user_ids = Variable(x[:,0])
        i1 = Variable(x[:,1])
        i2 = Variable(x[:,2])

        user_embedding = self.user_embeddings(user_ids)
        item1_embedding = self.item_embeddings(i1)
        item2_embedding = self.item_embeddings(i2)

        user_embedding = user_embedding.squeeze()
        item1_embedding = item1_embedding.squeeze()
        item2_embedding = item2_embedding.squeeze()

        x = torch.cat((user_embedding, item1_embedding, item2_embedding), 1)

        # 3 Layer
        # h_relu = self.linear1(x).clamp(min=0)
        # h_relu2 = self.linear2(h_relu).clamp(min=0)
        # h_relu3 = self.linear3(h_relu2).clamp(min=0)
        # drop = self.dropout(h_relu3)
        # score = self.linear4(drop)
        # prob = self.output(score)

        # 2 Layer
        # h_relu = self.linear1(x).clamp(min=0)
        # h_relu_bn = self.bn1(h_relu)
        # drop = self.dropout(h_relu_bn)
        # score = self.linear2(drop)
        # prob = self.output(score)

        h_relu1 = self.linear1(x).clamp(min=0)
        h_relu_bn1 = self.bn1(h_relu1)
        h_relu2 = self.linear2(h_relu_bn1).clamp(min=0)
        h_relu_bn2 = self.bn2(h_relu2)
        drop = self.dropout(h_relu_bn2)
        score = self.linear3(drop)
        prob = self.output(score)

        return prob



        # user_bias = self.user_biases(user_ids).squeeze()
        # item_bias = self.item_biases(item_ids).squeeze()
        #
        # dot = (user_embedding * item_embedding).sum(1)
        #
        # return dot + user_bias + item_bias
