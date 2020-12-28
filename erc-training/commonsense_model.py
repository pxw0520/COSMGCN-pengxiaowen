import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math
from model import SimpleAttention, MatchingAttention, Attention

class CommonsenseRNNCell(nn.Module):

    def __init__(self, D_m, D_s, D_g, D_p, D_r, D_i, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.5, emo_gru=True):    # emo_gru=True
        super(CommonsenseRNNCell, self).__init__()

        self.D_m = D_m
        self.D_s = D_s
        self.D_g = D_g
        self.D_p = D_p
        self.D_r = D_r
        self.D_i = D_i
        self.D_e = D_e

        # print ('dmsg', D_m, D_s, D_g)
        self.g_cell = nn.GRUCell(D_m+D_p+D_r, D_g)
        self.p_cell = nn.GRUCell(D_s+D_g, D_p)
        self.r_cell = nn.GRUCell(D_m+D_s+D_g, D_r)
        self.i_cell = nn.GRUCell(D_s+D_p, D_i)
        self.e_cell = nn.GRUCell(D_m+D_p+D_r+D_i, D_e)
        
        
        self.emo_gru = emo_gru
        self.listener_state = listener_state
        if listener_state:
            self.pl_cell = nn.GRUCell(D_s+D_g, D_p)
            self.rl_cell = nn.GRUCell(D_m+D_s+D_g, D_r)

        self.dropout = nn.Dropout(dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        if context_attention=='simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, U, x1, x2, x3, o1, o2, qmask, g_hist, q0, r0, i0, e0):
        """
        U -> batch, D_m
        x1, x2, x3, o1, o2 -> batch, D_m
        x1 -> effect on self; x2 -> reaction of self; x3 -> intent of self
        o1 -> effect on others; o2 -> reaction of others
        qmask -> batch, party       # 用户是谁
        g_hist -> t-1, batch, D_g  g_hist: i_{A,t-1}
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)
        # q_{A,t-1}
        q0_sel = self._select_parties(q0, qm_idx)
        r0_sel = self._select_parties(r0, qm_idx)

        ## global state ## g_hist: i_{A,t-1}
        g_ = self.g_cell(torch.cat([U, q0_sel, r0_sel], dim=1),
                torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0 else
                g_hist[-1])
        # g_ = self.dropout(g_)
        
        ## context ##
        if g_hist.size()[0]==0:
            c_ = torch.zeros(U.size()[0], self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist, U)
       
        ## external state ##
        U_r_c_ = torch.cat([U, x2, c_], dim=1).unsqueeze(1).expand(-1, qmask.size()[1],-1)
        # print ('urc', U_r_c_.size())
        # print ('u x2, c', U.size(), x2.size(), c_.size())
        # r_speaker
        rs_ = self.r_cell(U_r_c_.contiguous().view(-1, self.D_m+self.D_s+self.D_g),
                r0.view(-1, self.D_r)).view(U.size()[0], -1, self.D_r)
        # rs_ = self.dropout(rs_)
        
        ## internal state ##
        es_c_ = torch.cat([x1, c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = self.p_cell(es_c_.contiguous().view(-1, self.D_s+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
        # qs_ = self.dropout(qs_)
        

        if self.listener_state:
            ## listener external state ##
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
            er_ = o2.unsqueeze(1).expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_s)   # o2: other reaction
            ss_ = self._select_parties(rs_, qm_idx).unsqueeze(1).\
                    expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_r)
            U_er_ss_ = torch.cat([U_, er_, ss_], 1)
            rl_ = self.rl_cell(U_er_ss_, r0.view(-1, self.D_r)).view(U.size()[0], -1, self.D_r)
            # rl_ = self.dropout(rl_)
            
            ## listener internal state ##
            es_ = o1.unsqueeze(1).expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_s)   # o1: other effect
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_p)
            es_ss_ = torch.cat([es_, ss_], 1)
            ql_ = self.pl_cell(es_ss_, q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
            # ql_ = self.dropout(ql_)
            
        else:
            rl_ = r0
            ql_ = q0
            
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        r_ = rl_*(1-qmask_) + rs_*qmask_            
        
        ## intent ##        
        i_q_ = torch.cat([x3, self._select_parties(q_, qm_idx)], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)
        is_ = self.i_cell(i_q_.contiguous().view(-1, self.D_s+self.D_p),
                i0.view(-1, self.D_i)).view(U.size()[0], -1, self.D_i)
        # is_ = self.dropout(is_)
        il_ = i0
        i_ = il_*(1-qmask_) + is_*qmask_
        
        ## emotion ##        
        es_ = torch.cat([U, self._select_parties(q_, qm_idx), self._select_parties(r_, qm_idx), 
                         self._select_parties(i_, qm_idx)], dim=1)
        # 这个应该是e_{t-1}
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
                else e0
        
        if self.emo_gru:    # emo_gru = True
            e_ = self.e_cell(es_, e0)
        else:
            e_ = es_    
        
        # e_ = self.dropout(e_)
        g_ = self.dropout1(g_)
        q_ = self.dropout2(q_)
        r_ = self.dropout3(r_)
        i_ = self.dropout4(i_)
        e_ = self.dropout5(e_)
        
        return g_, q_, r_, i_, e_, alpha


class CommonsenseRNN(nn.Module):

    def __init__(self, D_m, D_s, D_g, D_p, D_r, D_i, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.5, emo_gru=True):
        super(CommonsenseRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_r = D_r
        self.D_i = D_i
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = CommonsenseRNNCell(D_m, D_s, D_g, D_p, D_r, D_i, D_e,
                            listener_state, context_attention, D_a, dropout, emo_gru)

    def forward(self, U, x1, x2, x3, o1, o2, qmask):
        """
        U -> seq_len, batch, D_m
        x1, x2, x3, o1, o2 -> seq_len, batch, D_s
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type()) # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_p).type(U.type()) # batch, party, D_p
        r_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_r).type(U.type()) # batch, party, D_r
        i_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_i).type(U.type()) # batch, party, D_i
        
        e_ = torch.zeros(0).type(U.type()) # batch, D_e
        e = e_

        alpha = []
        for u_, x1_, x2_, x3_, o1_, o2_, qmask_ in zip(U, x1, x2, x3, o1, o2, qmask):
            g_, q_, r_, i_, e_, alpha_ = self.dialogue_cell(u_, x1_, x2_, x3_, o1_, o2_, 
                                                            qmask_, g_hist, q_, r_, i_, e_)
            
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
            e = torch.cat([e, e_.unsqueeze(0)],0)
            
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        return e, alpha # seq_len, batch, D_e


class CommonsenseGRUModel(nn.Module):

    def __init__(self, D_m, D_s, D_g, D_p, D_r, D_i, D_e, D_h, D_a=100, n_classes=7, listener_state=False, 
        context_attention='simple', dropout_rec=0.5, dropout=0.1, emo_gru=True, mode1=0, norm=0, residual=False):

        super(CommonsenseGRUModel, self).__init__()

        if mode1 == 0:
            D_x = 4 * D_m
        elif mode1 == 1:
            D_x = 2 * D_m
        else:
            D_x = D_m

        self.mode1 = mode1
        self.norm_strategy = norm
        self.linear_in = nn.Linear(D_x, D_h)
        self.residual = residual

        self.r_weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))

        norm_train = True
        self.norm1a = nn.LayerNorm(D_m, elementwise_affine=norm_train)
        self.norm1b = nn.LayerNorm(D_m, elementwise_affine=norm_train)
        self.norm1c = nn.LayerNorm(D_m, elementwise_affine=norm_train)
        self.norm1d = nn.LayerNorm(D_m, elementwise_affine=norm_train)

        self.norm3a = nn.BatchNorm1d(D_m, affine=norm_train)
        self.norm3b = nn.BatchNorm1d(D_m, affine=norm_train)
        self.norm3c = nn.BatchNorm1d(D_m, affine=norm_train)
        self.norm3d = nn.BatchNorm1d(D_m, affine=norm_train)

        self.dropout   = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout_rec)
        self.cs_rnn_f = CommonsenseRNN(D_h, D_s, D_g, D_p, D_r, D_i, D_e, listener_state,
                                       context_attention, D_a, dropout_rec, emo_gru)
        self.cs_rnn_r = CommonsenseRNN(D_h, D_s, D_g, D_p, D_r, D_i, D_e, listener_state,
                                       context_attention, D_a, dropout_rec, emo_gru)
        self.sense_gru = nn.GRU(input_size=D_s, hidden_size=D_s//2, num_layers=1, bidirectional=True)
        self.matchatt = MatchingAttention(2*D_e,2*D_e,att_type='general2')
        self.linear     = nn.Linear(2*D_e, D_h)
        self.smax_fc    = nn.Linear(D_h, n_classes)

        # add
        self.linear_emotion = nn.Linear(6 * D_i, 200)

        max_seq_len = 110
        self.no_cuda = True
        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)
        return pad_sequence(xfs)

    def forward(self, r1, r2, r3, r4, x1, x2, x3, o1, o2, qmask, umask, att2=False, return_hidden=False):
        # 代码 train_or_eval_model 中 att2=True
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        seq_len, batch, feature_dim = r1.size()

        # default=3
        if self.norm_strategy == 1:
            # r1 = self.norm1a(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            # r2 = self.norm1b(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            # r3 = self.norm1c(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            # r4 = self.norm1d(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            raise ValueError

        elif self.norm_strategy == 2:
            # norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
            # r1 = norm2(r1.transpose(0, 1)).transpose(0, 1)
            # r2 = norm2(r2.transpose(0, 1)).transpose(0, 1)
            # r3 = norm2(r3.transpose(0, 1)).transpose(0, 1)
            # r4 = norm2(r4.transpose(0, 1)).transpose(0, 1)
            raise ValueError

        elif self.norm_strategy == 3:
            r1 = self.norm3a(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.norm3b(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.norm3c(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.norm3d(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)

        # default=2
        if self.mode1 == 0:
            # r = torch.cat([r1, r2, r3, r4], axis=-1)
            raise ValueError
        elif self.mode1 == 1:
            # r = torch.cat([r1, r2], axis=-1)
            raise ValueError
        elif self.mode1 == 2:
            r = (r1 + r2 + r3 + r4)/4
        # elif self.mode1 == 3:
        #     r = r1
        # elif self.mode1 == 4:
        #     r = r2
        # elif self.mode1 == 5:
        #     r = r3
        # elif self.mode1 == 6:
        #     r = r4
        # elif self.mode1 == 7:
        #     r = self.r_weights[0]*r1 + self.r_weights[1]*r2 + self.r_weights[2]*r3 + self.r_weights[3]*r4
        else:
            raise ValueError
            
        r = self.linear_in(r)   # in1024  out100
        
        emotions_f, alpha_f = self.cs_rnn_f(r, x1, x2, x3, o1, o2, qmask)
        
        out_sense, _ = self.sense_gru(x1)
        
        rev_r = self._reverse_seq(r, umask)
        rev_x1 = self._reverse_seq(x1, umask)
        rev_x2 = self._reverse_seq(x2, umask)
        rev_x3 = self._reverse_seq(x3, umask)
        rev_o1 = self._reverse_seq(o1, umask)
        rev_o2 = self._reverse_seq(o2, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.cs_rnn_r(rev_r, rev_x1, rev_x2, rev_x3, rev_o1, rev_o2, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        
        emotions = torch.cat([emotions_f,emotions_b],dim=-1)
        emotions = self.dropout_rec(emotions)
<<<<<<< Updated upstream
=======

        # --------------------------------------------------
        '''这里加GCN / FC: '''
        # emotions = self.linear_emotion(emotions)  # 900 -> 200

        self.window_past = 10
        self.window_future = 10
        n_speakers = 2
        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping
        # self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions, qmask, lengths,
                                                                                        self.window_past,
                                                                                        self.window_future,
                                                                                        self.edge_type_mapping,
                                                                                        self.att_model, self.no_cuda)

        #--------------------------------------------------
>>>>>>> Stashed changes
        
        alpha, alpha_f, alpha_b = [], [], []

        # 代码 train_or_eval_model：  att2=True
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
            
        hidden = self.dropout(hidden)
        
        if self.residual:
            hidden = hidden + r
        
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)

        if return_hidden:
            return hidden, alpha, alpha_f, alpha_b, emotions
        return log_prob, out_sense, alpha, alpha_f, alpha_b, emotions


def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    """
    Method to prepare the data format required for the GCN network. Pytorch geometric puts all nodes for classification
    in one single graph. Following this, we create a single graph for a mini-batch of dialogue instances. This method
    ensures that the various graph indexing is properly carried out so as to make sure that, utterances (nodes) from
    each dialogue instance will have edges with utterances in that same dialogue instance, but not with utternaces
    from any other dialogue instances in that mini-batch.
    """

    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))

    # scores are the edge weights
    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0] + length_sum, item[1] + length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))

        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])

            speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()

            if item1[0] < item1[1]:
                # edge_type.append(0) # ablation by removing speaker dependency: only 2 relation types
                # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0']) # ablation by removing temporal dependency: M^2 relation types
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                # edge_type.append(1) # ablation by removing speaker dependency: only 2 relation types
                # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0']) # ablation by removing temporal dependency: M^2 relation types
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    # if torch.cuda.is_available():
    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def edge_perms(l, window_past, window_future):
    """
    Method to construct the edges considering the past and future window.
    """

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j + window_future + 1)]
        elif window_future == -1:
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper.
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        """
        attn_type = 'attn1'

        if attn_type == 'attn1':

            scale = self.scalar(M)
            # scale = torch.tanh(scale)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)

            # if torch.cuda.is_available():
            if not self.no_cuda:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach().cuda()
                mask_copy = Variable(torch.zeros(alpha.size())).detach().cuda()

            else:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach()
                mask_copy = Variable(torch.zeros(alpha.size())).detach()

            edge_ind_ = []
            for i, j in enumerate(edge_ind):
                for x in j:
                    edge_ind_.append([i, x[0], x[1]])

            edge_ind_ = np.array(edge_ind_).transpose()
            mask[edge_ind_] = 1
            mask_copy[edge_ind_] = 1
            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True)
            scores = masked_alpha.div(_sums) * mask_copy
            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            # if torch.cuda.is_available():
            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            # if torch.cuda.is_available():
            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores