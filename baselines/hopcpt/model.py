import torch
import math
from hflayers import Hopfield


class HopfieldNet(torch.nn.Module):

    def __init__(self, dim_feature, dim_context_encoding, dim_hopfield_hidden, beta, use_temporal_encoding=False):
        super().__init__()
        
        self.dim_feature = dim_feature
        self.dim_context_encoding = dim_context_encoding
        self.dim_hopfield_hidden = dim_hopfield_hidden
        self.beta = beta
        self.use_temporal_encoding = use_temporal_encoding
        temporal_dim = 1 if self.use_temporal_encoding else 0
        hopfield_input_size = self.dim_context_encoding + temporal_dim
        hopfield_hidden_size = self.dim_hopfield_hidden + temporal_dim

        # no hidden layer following the original publication
        self.context_encoder = torch.nn.Sequential(
            torch.nn.Linear(dim_feature, self.dim_context_encoding),
        )
        
        self.hopfield_net = Hopfield(
            batch_first=True,
            input_size=hopfield_input_size,
            hidden_size=hopfield_hidden_size,
            output_size=None,
            stored_pattern_size=hopfield_input_size,
            pattern_projection_size=1,
            pattern_size=1,                      # value size (epsilon)
            num_heads=1,
            scaling=self._get_hopfield_scaling(hopfield_input_size),  # ~ beta (temperature), beta must be float
            disable_out_projection=True,         # keep head outputs separate
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,
            )

    def forward(self, k, v, association_mask):
        """
        model forward is only used for training 
        memory is used as query as well

        :param k: (batch_size, memory_length, dim_feature)
        :param v: (batch_size, memory_length, 1)
        :param association_mask: (batch_size, memory_length, memory_length)
        """
        device = k.device
        batch_size, memory_length, _ = k.shape

        if v.ndim == 1:
            v.unsqueeze_(0) 

        if v.ndim == 2:
            v.unsqueeze_(-1)

        # encoding
        encoded_k = self.context_encoder(k) # (batch_size, memory_length, dim_context_encoding)
        encoded_k = self._append_temporal_encoding(encoded_k, memory_length)

        return self.hopfield_net((encoded_k, encoded_k, v), association_mask=association_mask) # (batch_size, memory_length, 1)

    def obtain_association_matrix(self, k, q):
        """
        return association matrix for quantile estimation

        :param k: (batch_size, memory_length, dim_feature)
        :param q: (batch_size, 1, dim_feature)
        """
        device = k.device
        batch_size, memory_length, _ = k.shape

        # get_association_matrix() does not need value to compute the association matrix but needs value as an argument
        # this is dummy value
        v = torch.zeros((batch_size, memory_length, 1)).to(device)

        # encoding
        encoded_k = self.context_encoder(k) # (batch_size, memory_length, dim_context_encoding)
        encoded_k = self._append_temporal_encoding(encoded_k, memory_length)
        encoded_q = self.context_encoder(q) # (batch_size, 1, dim_context_encoding)
        encoded_q = self._append_temporal_encoding(encoded_q, memory_length, query=True)

        return self.hopfield_net.get_association_matrix((encoded_k, encoded_q, v)) # (batch_size, 1, 1, memory_length)

    def _append_temporal_encoding(self, encoded_context, memory_length, query=False):
        if not self.use_temporal_encoding:
            return encoded_context

        batch_size, sequence_length, _ = encoded_context.shape
        device = encoded_context.device
        if query:
            t = torch.ones((batch_size, sequence_length, 1), device=device)
        else:
            t = torch.arange(sequence_length, device=device)
            t = t / memory_length
            t = t.view(1, sequence_length, 1)
            t = t.expand(batch_size, -1, -1)
        return torch.cat([encoded_context, t], dim=-1)

    def _get_hopfield_scaling(self, dim):

        return self.beta / math.sqrt(dim)
