import torch, os, logging, random
from datetime import datetime
import numpy as np
from parse import parse_args

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def pLog(s: str):
    logging.info(s)

CORES = 16
os.environ['NUMEXPR_MAX_THREADS'] = '16' # mute warnings of logger
DATA_PATH = '../data/processed'
FILE_PATH = './checkpoints/'


ARG = parse_args()
LOG_FORMAT = "%(asctime)s  %(message)s"
DATE_FORMAT = "%m/%d %H:%M"


def _fmt_tag_val(v):
    if isinstance(v, float):
        s = f"{v:g}"
    else:
        s = str(v)
    return s.replace('-', 'm')


def _resolve_log_file():
    if ARG.log is not None:
        return ARG.log
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    geo_mode = "gttt" if ARG.use_geo_ttt else "gattn"
    auto_name = (
        f"{ARG.dataset.lower()}_"
        f"h{_fmt_tag_val(ARG.hidden)}_"
        f"tlr{_fmt_tag_val(ARG.ttt_base_lr)}_"
        f"tmb{_fmt_tag_val(ARG.ttt_mini_batch)}_"
        f"th{_fmt_tag_val(ARG.ttt_num_heads)}_"
        f"f{_fmt_tag_val(ARG.seq_logit_weight)}s{_fmt_tag_val(ARG.geo_logit_weight)}g_"
        f"{geo_mode}_{ts}.log"
    )
    return os.path.join("./logs", auto_name)


def _setup_dual_logging(log_file):
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


LOG_FILE = _resolve_log_file()
_setup_dual_logging(LOG_FILE)
logging.info(f"Logging to file: {LOG_FILE}")

SAVE = ARG.save
LOAD = ARG.load
SEED = ARG.seed
BATCH_SZ = ARG.batch
TEST_BATCH_SZ = ARG.testbatch
EPOCH = ARG.epoch
PATH = ARG.path
dataset = ARG.dataset
patience = ARG.patience

seed_torch(SEED)
os.makedirs(FILE_PATH, exist_ok=True)

dist_mat = torch.from_numpy(np.load(os.path.join(DATA_PATH, dataset.lower(), 'dist_mat.npy')))
device = torch.device('cpu' if ARG.gpu is None else f'cuda:{ARG.gpu}')
conf = {'lr': ARG.lr, 'decay': ARG.decay, 'num_layer': ARG.layer, 'hidden': ARG.hidden,
        'dropout': ARG.dropout, 'eval_all': ARG.eval_all, 'keepprob': ARG.keepprob, 'max_len': ARG.length,
        'interval': ARG.interval, 'T': ARG.diffsize, 'beta': ARG.beta, 'dt': ARG.stepsize,
        'use_geo_ttt': ARG.use_geo_ttt,
        'ttt_base_lr': ARG.ttt_base_lr, 'ttt_mini_batch': ARG.ttt_mini_batch,
        'ttt_num_heads': ARG.ttt_num_heads, 'ttt_num_hidden_layers': ARG.ttt_num_hidden_layers,
        'ttt_rope_theta': ARG.ttt_rope_theta,
        'seq_logit_weight': ARG.seq_logit_weight, 'geo_logit_weight': ARG.geo_logit_weight}

logging.info(
    f"Run config: hidden={ARG.hidden}, ttt(lr={ARG.ttt_base_lr:g}, mb={ARG.ttt_mini_batch}, heads={ARG.ttt_num_heads}, "
    f"layers={ARG.ttt_num_hidden_layers}), geo_mode={'TTT' if ARG.use_geo_ttt else 'Attn'}, "
    f"fuse={ARG.seq_logit_weight:g}*seq+{ARG.geo_logit_weight:g}*geo"
)
