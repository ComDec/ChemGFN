#!/bin/bash
# Launch 4 AMP experiments on GPU 4-7, one per GPU
# TB (GPU4), SubTB (GPU5), RapTB (GPU6), RapTB+SubM (GPU7)

set -e

COMMON="trainer.max_steps=5000 trainer.limit_train_batches=500 trainer.limit_val_batches=50"

echo "=== Launching AMP experiments on GPU 4-7 ==="

# GPU 4: TB
CUDA_VISIBLE_DEVICES=4 nohup python chemgfn/train.py \
  experiment=AMP/AMP_cfg_TB \
  $COMMON \
  > logs/amp_tb.log 2>&1 &
echo "TB       -> PID $! on GPU 4"

# GPU 5: SubTB
CUDA_VISIBLE_DEVICES=5 nohup python chemgfn/train.py \
  experiment=AMP/AMP_cfg_SubTB \
  $COMMON \
  > logs/amp_subtb.log 2>&1 &
echo "SubTB    -> PID $! on GPU 5"

# GPU 6: RapTB
CUDA_VISIBLE_DEVICES=6 nohup python chemgfn/train.py \
  experiment=AMP/AMP_cfg_RapTB \
  $COMMON \
  > logs/amp_raptb.log 2>&1 &
echo "RapTB    -> PID $! on GPU 6"

# GPU 7: RapTB+SubM
CUDA_VISIBLE_DEVICES=7 nohup python chemgfn/train.py \
  experiment=AMP/AMP_cfg_RapTB_SubM \
  $COMMON \
  > logs/amp_raptb_subm.log 2>&1 &
echo "RapTB+SubM -> PID $! on GPU 7"

echo ""
echo "All 4 jobs launched. Monitor with:"
echo "  tail -f logs/amp_tb.log"
echo "  tail -f logs/amp_subtb.log"
echo "  tail -f logs/amp_raptb.log"
echo "  tail -f logs/amp_raptb_subm.log"
echo ""
echo "Validation metrics to watch:"
echo "  val/topk_performance  (paper: Performance)"
echo "  val/topk_diversity    (paper: Diversity)"
echo "  val/topk_novelty      (paper: Novelty)"
