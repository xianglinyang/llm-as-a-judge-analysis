#! /bin/bash
input_files=(
    "/data2/xianglin/data/preference_leakage/UltraFeedback_sampled_30000_gemini_LlamaFactory.json"
    "/data2/xianglin/data/preference_leakage/UltraFeedback_sampled_30000_gpt4_LlamaFactory.json"
    "/data2/xianglin/data/preference_leakage/UltraFeedback_sampled_30000_llama_LlamaFactory.json"
)
output_files=(
    "/data2/xianglin/data/preference_leakage/output/gemini.json"
    "/data2/xianglin/data/preference_leakage/output/gpt4.json"
    "/data2/xianglin/data/preference_leakage/output/llama.json"
)

# Parallel execution
for i in ${!input_files[@]}; do
    python -m src.extract_features --input_file ${input_files[$i]} --output_file ${output_files[$i]} &
done
wait
