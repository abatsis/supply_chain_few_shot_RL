find ./data/test -name *.npz   | \
CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python evaluate.py ./models/latest {} >/dev/null 2>/dev/null"
