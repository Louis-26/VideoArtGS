# default parameters
USE_MULTI=1
KEEP_LOGS=1
MODE=1
output_dir="outputs"
save_dir="orig"

# read the parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use_multi) USE_MULTI="$2"; shift ;;
        --keep_logs) KEEP_LOGS="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --save_dir) SAVE_DIR="$2"; shift ;;
        *) echo "❌ Error: Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

JOB1_ID=$(sbatch --parsable init_cano.sh \
                --use_multi "$USE_MULTI" \
                --keep_logs "$KEEP_LOGS" \
                --mode "$MODE" \
                --output_dir "$OUTPUT_DIR")
echo "init_cano submitted with ID: $JOB1_ID"

JOB2_ID=$(sbatch --parsable --dependency=afterok:$JOB1_ID init_deform.sh \
                --use_multi "$USE_MULTI" \
                --keep_logs "$KEEP_LOGS" \
                --mode "$MODE" \
                --output_dir "$OUTPUT_DIR")
echo "init_deform submitted with ID: $JOB2_ID"

JOB3_ID=$(sbatch --parsable --dependency=afterok:$JOB2_ID ./train.sh \
                --use_multi "$USE_MULTI" \
                --keep_logs "$KEEP_LOGS" \
                --mode "$MODE" \
                --output_dir "$OUTPUT_DIR")
echo "train submitted with ID: $JOB3_ID"

JOB4_ID=$(sbatch --parsable --dependency=afterok:$JOB3_ID ./render.sh \
                --use_multi 0 \
                --keep_logs "$KEEP_LOGS" \
                --mode "$MODE" \
                --output_dir "$OUTPUT_DIR")
echo "render submitted with ID: $JOB4_ID"

JOB5_ID=$(sbatch --parsable --dependency=afterok:$JOB4_ID ./eval.sh \
                --use_multi "$USE_MULTI" \
                --keep_logs "$KEEP_LOGS" \
                --mode "$MODE" \
                --output_dir "$OUTPUT_DIR" \
                --save_dir "$SAVE_DIR")
echo "eval submitted with ID: $JOB5_ID"