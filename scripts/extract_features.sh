#! /bin/bash
input_files=(
    "/data2/xianglin/data/preference_leakage/UltraFeedback_sampled_30000_gemini_LlamaFactory.json"
    # "/data2/xianglin/data/preference_leakage/UltraFeedback_sampled_30000_gpt4_LlamaFactory.json"
    # "/data2/xianglin/data/preference_leakage/UltraFeedback_sampled_30000_llama_LlamaFactory.json"
)
output_files=(
    "/data2/xianglin/data/preference_leakage/output/gemini_feature_vecs"
    # "/data2/xianglin/data/preference_leakage/output/gpt4_feature_vecs.npy"
    # "/data2/xianglin/data/preference_leakage/output/llama_feature_vecs.npy"
)


for i in ${!input_files[@]}; do
    python -m src.extract_features --input_file ${input_files[$i]} --output_file ${output_files[$i]}
done
