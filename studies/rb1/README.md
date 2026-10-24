# RB1 — clean EOV-MMP baseline, and what the iterative-enhancement module adds

Pre-registration: [`../prereg/preregistration.md`](../prereg/preregistration.md)
(A23, then A23a). It continues the pilot pre-registration in the EOV repo, frozen
at A22. Nothing here is the thesis method.

`logs/` — hashes, the hash-verification log, the move log, test outputs.
Reports and run outputs go here too. Commands, per run (Step 2/3, after approval):

```bash
export OMP_NUM_THREADS=8
COMMON="--arch metor-compat --max_epoch 5 --lr 1e-5 --batch_size 1 \
        --no_epoch_eval --loader_seed 3407 --path_AFLink output/ckpt/AFLink_epoch20.pth"

# R1: as released (novel included, per-class fusion), no IEM
python -m cli.train $COMMON --n_iter 0 --expect_n_iter 0 --ps rb1_R1 \
    --epoch_ckpt_dir studies/rb1/runs/R1 --mirror_dir "$PERSIST/R1"
# R2: the clean baseline
python -m cli.train $COMMON --n_iter 0 --expect_n_iter 0 --exclude_novel_train --ps rb1_R2 \
    --epoch_ckpt_dir studies/rb1/runs/R2 --mirror_dir "$PERSIST/R2"
# R3: R2 + the iterative enhancement module only (not CRE)
python -m cli.train $COMMON --n_iter 2 --expect_n_iter 2 --exclude_novel_train --ps rb1_R3 \
    --epoch_ckpt_dir studies/rb1/runs/R3 --mirror_dir "$PERSIST/R3"
# resume after a machine loss: add  --resume studies/rb1/runs/Rk/epoch_00N.pth
```

`--uniform_fusion` acts only at inference: pass it when evaluating R0u, R2, R3.
