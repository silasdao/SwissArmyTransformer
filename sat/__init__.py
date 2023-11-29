from .arguments import get_args, update_args_with_file
from .tokenization import get_tokenizer
from .model import AutoModel

try:
    from .training.deepspeed_training import training_main
except ModuleNotFoundError as e:
    if 'deepspeed' not in str(e):
        raise e
    from sat.helpers import print_rank0
    print_rank0('DeepSpeed Not Installed, you cannot import training_main from sat now.', level="WARNING")
