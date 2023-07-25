#!/bin/bash
set -euo pipefail

# All blank parameters default to train_param_defaults.yaml values
WORK_ROOT=
WORK_DIR=

JOB_QUEUE="aml-gpu.q@b1,aml-gpu.q@b2,aml-gpu.q@b3"
JOB_NAME=
JOB_REASON=
experiment_suffix=

#Training hyperparameters
activation_layer=decoder.blocks.3.mlp.0
train_data=/exp/ellenar/sparse_coding/whisper_activations_tiny/train/${activation_layer}/train.dbl
val_data=/exp/ellenar/sparse_coding/whisper_activations_tiny/val/${activation_layer}/val.dbl
n_dict_components=1536
l1_alpha=5e-4
lr=4e-4
batch_size=10
n_gpus_per_node=1
steps=1000
grad_acc_steps=1

#Logging
log_every=10
log_tb_every=10
val_every=25
save_every=100

. parse_options.sh || exit 1;
set -o pipefail


# EXPERIMENT SETUP
JOB_NAME=${JOB_NAME:-"train"}
WORK_ROOT=${WORK_ROOT:-/exp/$(whoami)/sparse_coding/train}
experiment_suffix=${experiment_suffix:-whisper_tiny_${activation_layer}_n_dict_components_${n_dict_components}_l1_alpha_${l1_alpha}}
WORK_DIR=${WORK_DIR:-${WORK_ROOT}/$(date +"%Y%m%d")_$experiment_suffix}
JOB_REASON="${JOB_REASON:-"Training Sparse Autoencoders"}"
model_out_dir=${WORK_DIR}/models

mkdir -p $model_out_dir && chmod g+w "$WORK_DIR"

# Copy code into workspace
rsync --quiet -avhzL ./* $WORK_DIR/code 
train_copy="$WORK_DIR/$(basename $train_data)"
val_copy="$WORK_DIR/$(basename $val_data)"
cp -pf $train_data "$train_copy"
cp -pf $val_data "$val_copy"


cat <<EOF >"${WORK_DIR}"/run.qsh
#!/bin/bash
#$ -cwd
#$ -j y
#$ -S ${CODE_DIR}/env/singularity.sh
#$ -q ${JOB_QUEUE}
#$ -N "${JOB_NAME}"
#$ -terse
#$ -w w
#$ -wd ${WORK_DIR}/code
#$ -l gpu=$n_gpus_per_node
#$ -pe smp $n_gpus_per_node
#$ -notify
#$ -p 600
#$ -o ${WORK_DIR}/run.log
#$ -sync n
set -e pipefail;

# job info
hostname && date
echo
echo "sge_job_id:  \${JOB_ID}"
echo "sge_queue:   \${JOB_QUEUE}"
echo "user:        \${USER}"
echo "reason:      ${JOB_REASON}"
echo "sge_tmp_dir: \${TMPDIR}"
echo "sge_request: \${REQUEST}"
echo "sge_wd:      \$(pwd)"
echo "pstree_pid:  \$\$"
echo

function cleanup {
  err=\$?
  /home/ellenar/git/aladdin3/scripts/cleanup.sh --workdir "${WORK_DIR}" --message "${JOB_REASON}" --exitcode "\$err"
}

trap cleanup EXIT INT QUIT TERM
echo "\$(date -u) starting \${JOB_ID}" >> ${WORK_DIR}/sge_job_id

if [[ ! -f ${WORK_DIR}/done_train ]]; then
  python3 -m sparse_coding.train.train \
    --expdir $WORK_DIR \
    --steps $steps \
    --train_data $train_copy \
    --val_data $val_copy \
    --batch_size $batch_size \
    --grad_acc_steps $grad_acc_steps \
    --log_every $log_every \
    --log_tb_every $log_tb_every \
    --val_every $val_every \
    --save_every $save_every \
    --lr $lr \
    --checkpoint_autoload \
    --checkpoint_out ${model_out_dir}/checkpoint.pt \
    --model_out ${model_out_dir}/model.ts \
    --activation_layer $activation_layer \
    --n_dict_components $n_dict_components \
    --l1_alpha $l1_alpha
      ret_val="\$?"
      case "\$ret_val" in
        0) touch ${WORK_DIR}/done_train ;;
        12) echo "successful preempt" && exit 12 ;;
        *) exit "\$ret_val" ;;
      esac
fi

EOF
chmod +x "${WORK_DIR}"/run.qsh
job_id=$(qsub "${WORK_DIR}/run.qsh")
qmakep $job_id
sleep 4
echo "$job_id launched"
