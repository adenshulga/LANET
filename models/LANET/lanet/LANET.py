import torch 
from torch import nn
import math

from ..transformer import Constants


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 num_types, 
                 emb_dim, 
                 nhead = 4, 
                 dropout = 0.2, 
                 activation = 'relu', 
                 batch_first = True, 
                 encoder_num_layers = 2,
                 layer_norm_eps = 1e-5,
                 bias = True
                 ):
        
        super(TransformerEncoder, self).__init__()

        # self.device = device
        self.num_types = num_types
        self.emb_dim = emb_dim

        self.cat_embedding = nn.Embedding(num_embeddings=num_types, embedding_dim=emb_dim)
        self.position_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / self.emb_dim) for i in range(self.emb_dim)], 
                                         device=torch.device('cuda:0'))
        
        # self.position_vec = torch.tensor([math.pow(1, 2.0 * (1 // 2) / self.emb_dim) for i in range(self.emb_dim)], 
        #                             device=torch.device('cuda:0'))
        
        print('SELF POSITION VEC: ', self.position_vec)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=2*emb_dim, 
            nhead=nhead, 
            dim_feedforward=2*emb_dim, 
            dropout=dropout,
            activation=activation, 
            batch_first=batch_first, 
            layer_norm_eps=layer_norm_eps,
            bias=bias)
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=encoder_num_layers)

        self.dropout = nn.Dropout(0.3)
        self.encoder_history = nn.Linear(2*emb_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def temporal_enc(self, time):
        # [100000,  12032033, 0, 0, 0] / [1, 10000, ]
        # [0,1, 10-5]
        # [100, 10-6, 10-7]
        result = time.unsqueeze(-1) / self.position_vec
        # result = time.unsqueeze(-1) 
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result 
    
    def forward(self, event_type, event_time, label_ind: list[int] = None):

        
        # torch.save(event_time, '/app/All_models/model_pred_and_gt/lanet_cleann_event_time')
        # torch.save(event_type, '/app/All_models/model_pred_and_gt/lanet_cleann_event_type')

        # print(event_time)

        non_pad_mask = get_non_pad_mask(event_time)
        temp_enc = self.temporal_enc(event_time) * non_pad_mask # batch_size x seq_len x d_model

        # event_type # batch_size x seq_len x num_types 
        # non_pad_mask # batch_size x seq_len x 1
        # event_time # batch_size x seq_len 

        b = event_time.shape[0] 
        s = event_type.shape[1]

        x_cat_emb = self.cat_embedding(torch.arange(self.num_types, device=torch.device('cuda:0'))).unsqueeze(0).expand(b*s, -1, -1)
        # bs x num_types x emb_dim
        aux_mask = (1 - torch.triu(torch.ones(s, s, device=event_type.device), diagonal=1).T).unsqueeze(2).expand(-1, -1, self.num_types).transpose(1, 0)
        # s x s x num_types
        x = (event_type.unsqueeze(1).expand(-1, s, -1, -1) * aux_mask.unsqueeze(0).expand(b, -1, -1, -1)).reshape(b*s, s, self.num_types)
        # b x s x s x num_types
        x_t_emb = torch.sum(temp_enc.unsqueeze(1).expand(-1, s, -1, -1).reshape(b*s, s, self.emb_dim).unsqueeze(-1).expand(-1, -1, -1, self.num_types).transpose(3, 2) * \
                            x.unsqueeze(-1), dim=1) # bs x num_types x emb_dim
        
        if label_ind:   
            x_t_emb[:, label_ind, :] = 0

        x_encoder_input = torch.cat([x_cat_emb, x_t_emb], dim=2) # bs x num_types x 2*emb_dim 

        # if inner:
        # torch.save(x_encoder_input, 'model_pred_and_gt/LANET_CLEANN')
        # torch.save(x_t_emb, 'model_pred_and_gt/LANET_CLEANN_t_emb')
        # torch.save(x_cat_emb, 'model_pred_and_gt/LANET_CLEANN_cat_emb')


        x_encoder_output = self.transformer_encoder(x_encoder_input)
        x_hist = self.dropout(x_encoder_output)
        x_hist = self.encoder_history(x_hist).squeeze(2).reshape(b, s, -1) # b x s x num_types
        x_hist = self.sigmoid(x_hist)

        return x_hist, non_pad_mask
        

# 1 0 ...
# 1,2 0 ...
# 1,2,3 0 ..
# 1,2,3,4 0..
# 1,2,3,4,5 0... 
# 1,2,3,4,5
# 1,2,3,4,5
# 1,2,3,4,5
# 1,2,3,4,5





