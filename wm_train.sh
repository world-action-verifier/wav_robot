SUITE="robomimic"
TASK="can"
conda activate ${SUITE}_env

python3 train_sailor.py \
  --configs cfg_dp_mppi ${SUITE} \
  --task "${SUITE}__${TASK}" \
  --num_exp_trajs 50 \
  --wandb_exp_name "wm_only" \
  --set train_dp_mppi_params.update_dp_every 999999 \
  --set train_dp_mppi_params.n_dp_train_itrs 0