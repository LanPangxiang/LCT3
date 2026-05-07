import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Graph ODE for recommendation")
    parser.add_argument('--dataset', type=str, default='foursquare', # foursquare denotes singapore dataset
                        help="available datasets: ['foursquare', 'gowalla', 'nyc', 'tky',"
                             "'tky_10filter','nyc_original','nyc_sample','tky_5filter_sthgcn',','nyc_sample_test','nyc_sample_test_bu']")
    parser.add_argument('--epoch', type=int, default=100,
                        help='training epoch')
    parser.add_argument('--batch', type=int, default=1024,
                        help="the batch size for training procedure")
    parser.add_argument('--testbatch', type=int, default=1024,
                        help="the batch size of users for testing")
    parser.add_argument('--length', type=int, default=100,
                        help="max sequence length")
    parser.add_argument('--beta', type=float, default=0.2,
                        help="fisher loss weight")
    parser.add_argument('--hidden', type=int, default=96,
                        help="node embedding size")
    parser.add_argument('--interval', type=int, default=256,
                        help="types of temporal and locational intervals")
    parser.add_argument('--layer', type=int, default=2,
                        help="layer num of GNN")
    parser.add_argument('--diffsize', type=int, default=1,
                        help="diffusion size T")
    parser.add_argument('--stepsize', type=float, default=0.01,
                        help="diffusion step size dt")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-3,
                        help="weight decay for l2 normalizaton")
    parser.add_argument('--dropout', action='store_true', default=False,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="dropout probalitity")
    parser.add_argument('--patience', type=int, default=10,
                        help="early stop patience")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help='path to save weights')
    parser.add_argument('--log', type=str, default=None,
                        help="log file path")
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--eval_all', action='store_true', default=False)
    parser.add_argument('--use_geo_ttt', action='store_true', default=True,
                        help='use TTT in geo branch; default is off (keep original geo cross-attention)')
    parser.add_argument('--ttt_base_lr', type=float, default=0.00001,
                        help='base learning rate used inside TTT module')
    parser.add_argument('--ttt_mini_batch', type=int, default=8,
                        help='TTT mini-batch size (chunk size inside sequence)')
    parser.add_argument('--ttt_num_heads', type=int, default=2,
                        help='number of heads in TTT module')
    parser.add_argument('--ttt_num_hidden_layers', type=int, default=1,
                        help='number of residual TTT layers')
    parser.add_argument('--ttt_rope_theta', type=float, default=10000.0,
                        help='RoPE theta for TTT module')
    parser.add_argument('--seq_logit_weight', type=float, default=1,
                        help='weight for seq branch logits: seq_pre @ poi_emb^T')
    parser.add_argument('--geo_logit_weight', type=float, default=1,
                        help='weight for geo branch logits: loc_pre @ loc_embs^T')
    parser.add_argument('--seed', type=int, default=9876,
                        help='random seed')
    parser.add_argument('--gpu', type=str, default=None,
                        help='training device')
    return parser.parse_args()
