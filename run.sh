
set -x

./selfplay \
    --model_path resources/ckpts/gen0000.pt \
    --out_dir selfplay_logs \
    --n_workers 16 \
    --n_threads 8 \
    --n_searches 1000 \
    --starting_index 0 \
    --max_games 2000
