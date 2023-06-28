import os
import torch
from absl import flags, logging

from utils import device, get_checkpoint_to_start_from

FLAGS = flags.FLAGS
flags.DEFINE_string("model_out", None, "path to where to save trained model")
flags.DEFINE_string("checkpoint", None, "path to loading saved checkpoint")
flags.DEFINE_string("checkpoint_out", None, "path to save checkpoint")
flags.DEFINE_boolean(
    "checkpoint_autoload", True, "if True start from latest checkpoint_out"
)
flags.DEFINE_integer("seed", 5, "random seed")

flags.DEFINE_string("train_data", None, "path to train dblx")
flags.DEFINE_string("val_data", None, "path to validation dblx")
flags.DEFINE_string("default_testset", None, "path to default testset dblx")
flags.DEFINE_string("expdir", None, "directory to write all experiment data to")

flags.DEFINE_integer(
    "batch_size", 1, "batch size, num parallel streams to train on at once"
)
flags.DEFINE_integer("steps", -1, "maximum number of train steps")
flags.DEFINE_integer(
    "grad_acc_steps", 1, "number batches to accumulate grads before stepping"
)
flags.DEFINE_float("lr", 4e-4, "learning rate")

flags.DEFINE_integer("dl_max_workers", 0, "maximum number of dataloader subprocesses")
flags.DEFINE_integer(
    "n_devices", None, "do not set; automatically detected from CUDA_VISIBLE_DEVICES"
)
flags.DEFINE_integer("node_rank", 0, "rank of current node in distributed environment")
flags.DEFINE_integer("n_nodes", 1, "total number of nodes in distributed environment")
# gloo required for sparse grads
flags.DEFINE_enum(
    "dist_backend", "nccl", ["nccl", "gloo"], "distributed training backend"
)

flags.DEFINE_integer("val_every", None, "perform validation every n steps")
flags.DEFINE_integer("save_every", None, "save every n steps")
flags.DEFINE_integer("log_every", 1, "append to log file every n steps")
flags.DEFINE_integer("log_tb_every", 50, "save tb scalars every n steps")
flags.DEFINE_integer("log_tb_viz_every", 500, "save vizualisations every n steps")

flags.mark_flag_as_required("save_every")
flags.mark_flag_as_required("val_every")
flags.mark_flag_as_required("train_data")
flags.mark_flag_as_required("val_data")
flags.mark_flag_as_required("expdir")


def train_init(train, unused_argv):
    """
    N.B. Wrap with partial when passing to app.run as that requires a single callable
    without additional arguments e.g. app.run(partial(train_init, train))
    """

    # alternative to torch.set_num_threads(1) which is propagated to spawned child processes
    # see https://github.com/pytorch/pytorch/issues/44025
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDNN_V8_API_ENABLED"] = "1"
    os.environ["USE_HEURISTIC_MODE_B"] = "1"

    # Set PyTorch JIT layer fusion options - nvfuser.
    # These settings are also set when using a context manager to set the fuser,
    # e.g. with torch.jit.fuser("fuser2"):
    # See JIT source code (https://enchanter.readthedocs.io/en/v0.8.0/_modules/torch/jit.html) for details.
    torch._C._jit_set_profiling_executor(
        True
    )  # nvFuser requires the profiling executor. It provides type information.
    torch._C._jit_set_profiling_mode(
        True
    )  # Not 100% necessary, but performance is terrible when False.
    torch._C._jit_override_can_fuse_on_cpu(
        False
    )  # Only True when using the legacy fuser.
    torch._C._jit_override_can_fuse_on_gpu(
        False
    )  # Only True when using the legacy fuser.
    torch._C._jit_set_texpr_fuser_enabled(
        False
    )  # This disables fuser1 (NNC, otherwise called the TensorExpr fuser).
    torch._C._jit_set_nvfuser_enabled(
        True
    )  # This sets the backend fuser to be fuser2, i.e. nvFuser
    torch._C._debug_set_autodiff_subgraph_inlining(
        False
    )  # Setting to True can be useful for debugging. See:
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/runtime/graph_executor.cpp#L82

    logging.info(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # covers single gpu case as torch already imported so env variables won't effect state
    torch.set_num_threads(1)

    if not FLAGS.model_out:
        models_dir = f"{FLAGS.expdir}/models"
        FLAGS.model_out = f"{models_dir}/model.pt"
        os.makedirs(models_dir, exist_ok=True)

    if not FLAGS.checkpoint_out:
        checkpoint_dir = f"{FLAGS.expdir}/checkpoints"
        FLAGS.checkpoint_out = f"{checkpoint_dir}/checkpoint.pt"
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Additional FLAG parsing
    if FLAGS.checkpoint_autoload is True:
        latest_checkpoint = get_checkpoint_to_start_from(FLAGS.checkpoint_out)
        # Override FLAGS.checkpoint to handle preemption edge case
        # if we cannot find an automatic checkpoint, default to user defined checkpoint
        if latest_checkpoint is not None:
            FLAGS.checkpoint = latest_checkpoint
        logging.info(f"autosetting checkpoint: {FLAGS.checkpoint}")

    if not FLAGS.save_every:
        FLAGS.save_every = FLAGS.val_every

    if device == torch.device("cpu"):
        FLAGS.n_devices = 1
    else:
        # TODO: remove .get fallback once we use torchrun everywhere.
        FLAGS.n_devices = int(
            os.environ.get("WORLD_SIZE", torch.cuda.device_count() * FLAGS.n_nodes)
        )

    FLAGS.node_rank = int(
        os.environ.get("GROUP_RANK", FLAGS.node_rank)
    )  # TODO: remove .get fallback once we use torchrun everywhere.
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    FLAGS.n_nodes = int(
        FLAGS.n_devices / device_count
    )  # assumes equal number of gpus per node

    train(FLAGS)
