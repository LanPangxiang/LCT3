import gol
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.nn.utils.rnn import pad_sequence

from layers import GeoConv, SeqConv

try:
    from ttt import TTTMLP
    _TTT_IMPORT_ERROR = None
except Exception as _ttt_err:
    TTTMLP = None
    _TTT_IMPORT_ERROR = _ttt_err


class TTTTransformerStack(nn.Module):
    """
    Transformer-style residual stack with TTTMLP sequence modeling blocks.
    This class intentionally keeps only the base TTT path (no hyperbolic/time-space control).
    """

    def __init__(
        self,
        hidden_size,
        num_heads=2,
        num_hidden_layers=1,
        mini_batch_size=8,
        ttt_hidden_dim=None,
        ttt_base_lr=0.1,
        rope_theta=1000.0,
        dropout=0.2,
    ):
        super().__init__()
        if TTTMLP is None:
            raise ImportError(
                "Failed to import TTTMLP from ttt.py. Please ensure ttt.py dependencies are installed."
            ) from _TTT_IMPORT_ERROR

        self.num_heads = int(num_heads)
        self.num_hidden_layers = int(num_hidden_layers)
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be >= 1")

        if ttt_hidden_dim is None:
            ttt_hidden_dim = int(hidden_size*2)
        ttt_hidden_dim = int(ttt_hidden_dim)
        if ttt_hidden_dim % self.num_heads != 0:
            ttt_hidden_dim = ((ttt_hidden_dim + self.num_heads - 1) // self.num_heads) * self.num_heads
        self.ttt_hidden_dim = ttt_hidden_dim

        self.in_proj = nn.Identity() if self.ttt_hidden_dim == hidden_size else nn.Linear(hidden_size, self.ttt_hidden_dim)
        self.out_proj = nn.Identity() if self.ttt_hidden_dim == hidden_size else nn.Linear(self.ttt_hidden_dim, hidden_size)

        ttt_config = SimpleNamespace(
            hidden_size=self.ttt_hidden_dim,
            num_attention_heads=self.num_heads,
            mini_batch_size=int(mini_batch_size),
            ttt_base_lr=float(ttt_base_lr),
            rope_theta=float(rope_theta),
            share_qk=False,
            use_gate=False,
            pre_conv=False,
            num_hidden_layers=self.num_hidden_layers,
            conv_kernel=4,
            rms_norm_eps=1e-6,
            scan_checkpoint_group_size=0,
            ttt_layer_type="mlp",
        )

        self.ttt_layers = nn.ModuleList([TTTMLP(ttt_config, layer_idx=i) for i in range(self.num_hidden_layers)])
        self.seq_norms = nn.ModuleList([nn.LayerNorm(self.ttt_hidden_dim) for _ in range(self.num_hidden_layers)])

        ffn_mult = 4
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(self.ttt_hidden_dim) for _ in range(self.num_hidden_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.ttt_hidden_dim, ffn_mult * self.ttt_hidden_dim),
                nn.SiLU(),
                nn.Linear(ffn_mult * self.ttt_hidden_dim, self.ttt_hidden_dim),
            ) for _ in range(self.num_hidden_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        # x: [B, L, C], pad_mask: [B, L] (True for valid tokens)
        if pad_mask is not None:
            x = x * pad_mask.to(dtype=x.dtype).unsqueeze(-1)

        h = self.in_proj(x)
        bsz, seq_len, _ = h.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=h.device).unsqueeze(0).expand(bsz, -1)

        for i, ttt_layer in enumerate(self.ttt_layers):
            residual = h
            h_norm = self.seq_norms[i](h)
            ttt_out = ttt_layer(h_norm, attention_mask=pad_mask, position_ids=position_ids)
            h = residual + self.dropout(ttt_out)

            residual = h
            h_ffn = self.ffn_norms[i](h)
            h = residual + self.dropout(self.ffns[i](h_ffn))

            if pad_mask is not None:
                h = h * pad_mask.to(dtype=h.dtype).unsqueeze(-1)

        out = self.out_proj(h)
        if pad_mask is not None:
            out = out * pad_mask.to(dtype=out.dtype).unsqueeze(-1)
        return out


class LaMDA(nn.Module):
    def __init__(self, n_user, n_poi, geo_graph: Data):
        super(LaMDA, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.hid_dim = gol.conf['hidden']
        self.step_num = 1000
        self.local_pois = 20

        self.poi_emb = nn.Parameter(torch.empty(n_poi, self.hid_dim))
        self.distance_emb = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        nn.init.xavier_normal_(self.poi_emb)
        nn.init.xavier_normal_(self.distance_emb)

        self.geo_encoder = GeoEncoder(n_poi, self.hid_dim, geo_graph)
        self.seq_encoder = SeqEncoder(self.hid_dim)
        self.ce_criteria = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=1 - gol.conf['keepprob'])

        ttt_num_heads = int(gol.conf.get('ttt_num_heads', 2))
        ttt_num_hidden_layers = int(gol.conf.get('ttt_num_hidden_layers', 1))
        ttt_mini_batch = int(gol.conf.get('ttt_mini_batch', 8))
        ttt_base_lr = float(gol.conf.get('ttt_base_lr', 0.1))
        ttt_rope_theta = float(gol.conf.get('ttt_rope_theta', 10000.0))
        self.use_geo_ttt = bool(gol.conf.get('use_geo_ttt', False))
        self.seq_logit_weight = float(gol.conf.get('seq_logit_weight', 1))
        self.geo_logit_weight = float(gol.conf.get('geo_logit_weight', 1))

        self.seq_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.seq_attn = TTTTransformerStack(
            hidden_size=self.hid_dim,
            num_heads=ttt_num_heads,
            num_hidden_layers=ttt_num_hidden_layers,
            mini_batch_size=ttt_mini_batch,
            ttt_hidden_dim=self.hid_dim,
            ttt_base_lr=ttt_base_lr,
            rope_theta=ttt_rope_theta,
            dropout=0.2,
        )

        self.geo_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        self.geo_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8)
        if self.use_geo_ttt:
            self.geo_attn = TTTTransformerStack(
                hidden_size=self.hid_dim,
                num_heads=ttt_num_heads,
                num_hidden_layers=ttt_num_hidden_layers,
                mini_batch_size=ttt_mini_batch,
                ttt_hidden_dim=self.hid_dim,
                ttt_base_lr=ttt_base_lr,
                rope_theta=ttt_rope_theta,
                dropout=0,
            )
        else:
            self.geo_attn = nn.MultiheadAttention(
                self.hid_dim,
                num_heads=2,
                batch_first=True,
                dropout=0.2,
            )

    def location_Pro(self, poi_embs, seqs, seq_pre):
        loc_embs = self.geo_encoder.encode(poi_embs)
        if gol.conf['dropout']:
            loc_embs = self.dropout(loc_embs)

        seq_lengths = torch.LongTensor([seq.size(0) for seq in seqs]).to(gol.device)
        geo_seq_embs = [loc_embs[seq] for seq in seqs]
        loc_embs_pad = pad_sequence(geo_seq_embs, batch_first=True, padding_value=0)

        qry_embs = self.geo_layernorm(seq_pre.detach().unsqueeze(1))
        pad_mask = sequence_mask(seq_lengths)
        if self.use_geo_ttt:
            qry_mask = torch.ones((pad_mask.size(0), 1), dtype=pad_mask.dtype, device=pad_mask.device)
            geo_inputs = torch.cat([qry_embs, loc_embs_pad], dim=1)
            geo_mask = torch.cat([qry_mask, pad_mask], dim=1)
            geo_outputs = self.geo_attn(geo_inputs, pad_mask=geo_mask)
            # Query token is the first token; keep [B, D] for downstream logits.
            # loc_pre = self.geo_attn_layernorm(geo_outputs[:, 0, :])
            loc_pre = geo_outputs[:, 0, :]
        else:
            loc_ctx, _ = self.geo_attn(
                qry_embs,
                loc_embs_pad,
                loc_embs_pad,
                key_padding_mask=~pad_mask,
            )
            loc_pre = self.geo_attn_layernorm(loc_ctx.squeeze(1))
        return loc_pre, loc_embs

    def sequence_Pro(self, poi_embs, seq_graph):
        seq_embs = self.seq_encoder.encode((poi_embs, self.distance_emb), seq_graph)
        if gol.conf['dropout']:
            seq_embs = self.dropout(seq_embs)

        seq_lengths = torch.bincount(seq_graph.batch)
        seq_embs = torch.split(seq_embs, seq_lengths.cpu().numpy().tolist())

        seq_embs_pad = pad_sequence(seq_embs, batch_first=True, padding_value=0)
        pad_mask = sequence_mask(seq_lengths)

        seq_inputs = self.seq_layernorm(seq_embs_pad)
        seq_out = self.seq_attn(seq_inputs, pad_mask=pad_mask)

        seq_out = [seq[:seq_len] for seq, seq_len in zip(seq_out, seq_lengths)]
        seq_pre = torch.stack([seq.mean(dim=0) for seq in seq_out], dim=0)
        return seq_pre, seq_embs

    def getTrainLoss(self, batch):
        usr, pos_lbl, _, seqs, seq_graph, cur_time = batch
        poi_embs = self.poi_emb
        if gol.conf['dropout']:
            poi_embs = self.dropout(poi_embs)

        seq_pre, seq_embs = self.sequence_Pro(poi_embs, seq_graph)
        loc_pre, loc_embs = self.location_Pro(poi_embs, seqs, seq_pre)
        pred_logits = self.seq_logit_weight * (seq_pre @ self.poi_emb.T) + self.geo_logit_weight * (loc_pre @ loc_embs.T)
        return self.ce_criteria(pred_logits, pos_lbl)

    def forward(self, seqs, seq_graph):
        poi_embs = self.poi_emb
        seq_pre, seq_embs = self.sequence_Pro(poi_embs, seq_graph)
        loc_pre, loc_embs = self.location_Pro(poi_embs, seqs, seq_pre)
        pred_logits = self.seq_logit_weight * (seq_pre @ self.poi_emb.T) + self.geo_logit_weight * (loc_pre @ loc_embs.T)
        return pred_logits


class SeqEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(SeqEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = SeqConv(hid_dim)

    def encode(self, embs, seq_graph):
        return self.encoder(embs, seq_graph)


class GeoEncoder(nn.Module):
    def __init__(self, n_poi, hid_dim, geo_graph: Data):
        super(GeoEncoder, self).__init__()
        self.n_poi, self.hid_dim = n_poi, hid_dim
        self.gcn_num = gol.conf['num_layer']

        edge_index, _ = add_self_loops(geo_graph.edge_index)
        dist_vec = torch.cat([geo_graph.edge_attr, torch.zeros((n_poi,)).to(gol.device)])
        dist_vec = torch.exp(-(dist_vec ** 2))
        self.geo_graph = Data(edge_index=edge_index, edge_attr=dist_vec)

        self.act = nn.LeakyReLU()
        self.geo_convs = nn.ModuleList()
        for _ in range(self.gcn_num):
            self.geo_convs.append(GeoConv(self.hid_dim, self.hid_dim))

    def encode(self, poi_embs):
        layer_embs = poi_embs
        loc_embs = [layer_embs]
        for conv in self.geo_convs:
            layer_embs = conv(layer_embs, self.geo_graph)
            layer_embs = self.act(layer_embs)
            loc_embs.append(layer_embs)
        loc_embs = torch.stack(loc_embs, dim=1).mean(1)
        return loc_embs


def sequence_mask(lengths, max_len=None):
    lengths_shape = lengths.shape
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)

