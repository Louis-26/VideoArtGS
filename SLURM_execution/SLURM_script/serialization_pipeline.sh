JOB1_ID=$(sbatch --parsable ./init_cano.sh 1)
echo "init_cano submitted with ID: $JOB1_ID"

JOB2_ID=$(sbatch --parsable --dependency=afterok:$JOB1_ID ./init_deform_PAT.sh 1)
echo "init_deform_PAT submitted with ID: $JOB2_ID"

JOB3_ID=$(sbatch --parsable --dependency=afterok:$JOB2_ID ./train.sh 1)
echo "init_deform_PAT submitted with ID: $JOB3_ID"

JOB4_ID=$(sbatch --parsable --dependency=afterok:$JOB3_ID ./render.sh 1)
echo "init_deform_PAT submitted with ID: $JOB4_ID"

JOB5_ID=$(sbatch --parsable --dependency=afterok:$JOB4_ID ./eval.sh 1)
echo "init_deform_PAT submitted with ID: $JOB5_ID"