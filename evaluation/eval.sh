#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # adjust according to your GPU configuration

export QWEN_LOCAL_URL="http://29.127.68.253:8000/v1/chat/completions"
export MODEL="Qwen/Qwen2.5-72B-Instruct"

# ========================
# Configuration: Pass@k Settings
# ========================

PASS_K_TEMPERATURE=0.6  # Used when n > 1 for diverse sampling
DATASETS="mathverse,mathvista,wemath,hallubench,chartqa,logicvista"
MAX_MODEL_LEN=16384
MAX_NUM_BATCHED_TOKENS=32768
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.9
PROMPT_WORKERS=16
JUDGE_MAX_CONCURRENT=256
OVERWRITE=true  # Set to true to re-run inference/evaluation even if results exist

# ========================
# Inference Stage:
# Get inference results for all the benchmarks
# ========================

HF_MODEL_PATHS=(

)

RESULTS_DIRS=(

)



SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer MUST BE put in \boxed{}."""

if [ "${#HF_MODEL_PATHS[@]}" -ne "${#RESULTS_DIRS[@]}" ]; then
  echo "HF_MODEL_PATHS and RESULTS_DIRS must have the same length" >&2
  exit 1
fi


for i in "${!HF_MODEL_PATHS[@]}"; do
  HF_MODEL_PATH="${HF_MODEL_PATHS[$i]}"
  RESULTS_DIR="${RESULTS_DIRS[$i]}"

  mkdir -p "$RESULTS_DIR"
  
  echo "Inferencing model: $HF_MODEL_PATH"
  echo "Inferencing results will be saved to: $RESULTS_DIR"
  
  # Detect if model is 3B and adjust tensor parallelism
  if [[ "$HF_MODEL_PATH" == *"3b"* ]] || [[ "$HF_MODEL_PATH" == *"3B"* ]]; then
    TENSOR_PARALLEL=1  # 3B models don't need tensor parallelism
    echo "Using tensor-parallel-size=1 for 3B model"
  else
    TENSOR_PARALLEL=4  # 7B and larger models benefit from parallelism
    echo "Using tensor-parallel-size=4 for larger model"
  fi
  
  OVERWRITE_FLAG=""
  if [ "$OVERWRITE" = true ]; then
    OVERWRITE_FLAG="--overwrite"
  fi

  python evaluation/inference.py \
    --model "$HF_MODEL_PATH" \
    --output-dir "$RESULTS_DIR" \
    --datasets "$DATASETS" \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --system-prompt="$SYSTEM_PROMPT" \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-seqs $MAX_NUM_SEQS \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --prompt-workers $PROMPT_WORKERS \
    --k 1 \
    --pass-k-temperature "$PASS_K_TEMPERATURE" \
    $OVERWRITE_FLAG

  echo "Finished inferencing $HF_MODEL_PATH"
  echo "-----------------------------------"
done


# ========================
# Evaluation Stage: 
# Evaluate the inference results for all the benchmarks
# ========================

for i in "${!HF_MODEL_PATHS[@]}"; do
  HF_MODEL_PATH="${HF_MODEL_PATHS[$i]}"
  RESULTS_DIR="${RESULTS_DIRS[$i]}"
  
  echo "Evaluating model: $HF_MODEL_PATH"
  echo "Evaluating results will be saved to: $RESULTS_DIR"
  
  OVERWRITE_FLAG=""
  if [ "$OVERWRITE" = true ]; then
    OVERWRITE_FLAG="--overwrite"
  fi

  python evaluation/evaluation.py \
    --datasets "$DATASETS" \
    --output-dir "$RESULTS_DIR" \
    --system-prompt="$SYSTEM_PROMPT" \
    --pass-k-values "1" \
    --max-concurrent $JUDGE_MAX_CONCURRENT \
    $OVERWRITE_FLAG
  
  echo "Finished evaluating $HF_MODEL_PATH"
  echo "-----------------------------------"
done

RESULTS_DIRS_CSV=""
MODEL_NAMES_CSV=""
for i in "${!RESULTS_DIRS[@]}"; do
  RESULTS_DIR="${RESULTS_DIRS[$i]}"
  HF_MODEL_PATH="${HF_MODEL_PATHS[$i]}"

  if [ -n "$RESULTS_DIRS_CSV" ]; then
    RESULTS_DIRS_CSV+=","
    MODEL_NAMES_CSV+=","
  fi
  RESULTS_DIRS_CSV+="$RESULTS_DIR"
  MODEL_NAMES_CSV+="$(basename "$RESULTS_DIR")"
done

echo "Aggregating all model results into one table"
python evaluation/aggregate_results.py \
  --results-dirs "$RESULTS_DIRS_CSV" \
  --model-names "$MODEL_NAMES_CSV" \
  --datasets "$DATASETS"

echo "All evaluations completed!"

python ../matrix_multiplication_gpus.py --gpus 8 --size 5000

