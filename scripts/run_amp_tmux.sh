#!/bin/bash
# Launch 4 AMP experiments in tmux on GPU 4-7
# Usage: bash scripts/run_amp_tmux.sh

cd /home/xiwang/project/ChemGFN

tmux new-session -d -s amp -n tb
tmux send-keys -t amp:tb "CUDA_VISIBLE_DEVICES=4 conda run -n chemgfn python chemgfn/train.py experiment=AMP/AMP_cfg_TB" Enter

tmux new-window -t amp -n subtb
tmux send-keys -t amp:subtb "CUDA_VISIBLE_DEVICES=5 conda run -n chemgfn python chemgfn/train.py experiment=AMP/AMP_cfg_SubTB" Enter

tmux new-window -t amp -n raptb
tmux send-keys -t amp:raptb "CUDA_VISIBLE_DEVICES=6 conda run -n chemgfn python chemgfn/train.py experiment=AMP/AMP_cfg_RapTB" Enter

tmux new-window -t amp -n raptb_subm
tmux send-keys -t amp:raptb_subm "CUDA_VISIBLE_DEVICES=7 conda run -n chemgfn python chemgfn/train.py experiment=AMP/AMP_cfg_RapTB_SubM" Enter

echo "tmux session 'amp' created. Attach with: tmux attach -t amp"
