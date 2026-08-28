# default parameters
USE_MULTI=1
KEEP_LOGS=1
MODE=1
OUTPUT_DIR="outputs_PAT"
SAVE_DIR="PAT"
PAT_MODEL_PTH="particulate/model_ckpt/pat_model.pt"
PAT_NUM_POINTS=65536

# read the parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use_multi) USE_MULTI="$2"; shift ;;
        --keep_logs) KEEP_LOGS="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --save_dir) SAVE_DIR="$2"; shift ;;
        --PAT_model_pth) PAT_MODEL_PTH="$2"; shift ;;
        --pat_num_points) PAT_NUM_POINTS="$2"; shift ;;
        *) echo "❌ Error: Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# stage 1, initialize the canonical Gaussian Primitives
# echo "🎬 Stage 1: Initialize the canonical Gaussian Primitives"
bash scripts/init_cano.sh \
    --use_multi "$USE_MULTI" \
    --keep_logs "$KEEP_LOGS" \
    --mode "$MODE" \
    --output_dir "$OUTPUT_DIR"

# stage 2, initialize the deformation field 
echo "🎬 Stage 2: Initialize the deformation field"
bash scripts/init_deform_PAT.sh \
    --use_multi "$USE_MULTI" \
    --keep_logs "$KEEP_LOGS" \
    --mode "$MODE" \
    --output_dir "$OUTPUT_DIR" \
    --PAT_model_pth "$PAT_MODEL_PTH" \
    --pat_num_points "$PAT_NUM_POINTS"

# stage 3, jointly train the canonical Gaussian Primitives and the deformation field
echo "🎬 Stage 3: Jointly train the canonical Gaussian Primitives and the deformation"
bash scripts/train_PAT.sh \
    --use_multi "$USE_MULTI" \
    --keep_logs "$KEEP_LOGS" \
    --mode "$MODE" \
    --output_dir "$OUTPUT_DIR"

# stage 4, 3D deformed gaussians rendering
echo "🎬 Stage 4: 3D deformed gaussians rendering"
bash scripts/render.sh \
    --use_multi 0 \
    --keep_logs "$KEEP_LOGS" \
    --mode "$MODE" \
    --output_dir "$OUTPUT_DIR"

bash scripts/render_mask.sh \
    --use_multi 0 \
    --keep_logs "$KEEP_LOGS" \
    --mode "$MODE" \
    --output_dir "$OUTPUT_DIR"

# stage 5, evaluation
echo "🎬 Stage 5: Evaluation"
bash scripts/eval.sh \
    --use_multi "$USE_MULTI" \
    --keep_logs "$KEEP_LOGS" \
    --mode "$MODE" \
    --output_dir "$OUTPUT_DIR" \
    --save_dir "$SAVE_DIR"