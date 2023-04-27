import torch
import torch.nn as nn
import numpy as np

class FeaturesLinear(nn.Module):
    """
    A module to create the linear part of Factorization Machines and Field-aware Factorization Machines.

    Parameters
    ----------
    field_dims : List[int]
        A list of field dimensions.
    """
    def __init__(self, field_dims):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the linear part of the model.

        Parameters
        ----------
        x : Long tensor of size (batch_size, num_fields)

        Returns
        ----------
        y : Float tensor of size (batch_size)
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        y = torch.sum(self.fc(x), dim=1) + self.bias
        return y
    
    
class FeaturesEmbedding(nn.Module):
    """
    A module to create feature embeddings.
    
    Parameters
    ----------
    field_dims : List[int]
        A list of field dimensions.
    embedding_dim : int
        The dimension of dense embeddings.
    """

    def __init__(self, field_dims, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embedding_dim)
        self.offsets = torch.tensor((0, *np.cumsum(field_dims)[:-1]), dtype=torch.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        Forward pass through the feature embeddings.

        Parameters
        ----------
        x : Long tensor of size (batch_size, num_fields)

        Returns
        -------
        y : Float tensor of size (batch_size, num_fields, embedding_dim)
        """
        x = x + self.offsets.unsqueeze(0).to(x.device)
        y = self.embedding(x)
        return y
    
    
class PairwiseInteraction(nn.Module):
    """
    A module to compute the pairwise interactions.

    Parameters
    ----------
    embedding_dim : int
        The dimension of dense embeddings.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass through the pairwise interactions.

        Parameters
        ----------
        x : Float tensor of size (batch_size, num_fields, embedding_dim)

        Returns
        -------
        y : Float tensor of size (batch_size)
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        y = 0.5 * (square_of_sum - sum_of_square)
        y = torch.sum(y, dim=1).unsqueeze(-1)
        return y



class MyFactorizationMachine(nn.Module):
    """
    A class implementing the Factorization Machine model.
    
    Parameters
    ----------
    field_dims : List[int]
        A list of field dimensions.
    embedding_dim : int
        The dimension of dense embeddings.
    """

    def __init__(self, args):
        super().__init__()
        self.field_dims = args['field_dims']
        self.embedding_dim = args['embed_dim']
        self.linear = FeaturesLinear(self.field_dims)
        self.embedding = FeaturesEmbedding(self.field_dims, self.embedding_dim)
        self.interaction = PairwiseInteraction()

    def forward(self, x):
        """
        Forward pass through the Factorization Machine model.

        Parameters
        ----------
        x : Long tensor of size (batch_size, num_fields)

        Returns
        ----------
        y : Float tensor of size (batch_size)
        """
        y_linear = self.linear(x)
        embed_x = self.embedding(x)
        y_interaction = self.interaction(embed_x)
        y = y_linear + y_interaction
        return y.squeeze(1)
