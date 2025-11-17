lm_eval --model steer_hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,\
steer_layer=20,steer_lambda=0.8,steer_span=1,\
steer_vec_path=/home/youyang7/projects/fact-enhancement/artifacts/factual_dirs.json \
  --tasks triviaqa,truthfulqa_gen \
  --num_fewshot 0 --apply_chat_template \
  --device cuda:1 --batch_size auto --limit 0.02 \
  --output_path ./eval_out/steer_demo --log_samples


lm_eval --model hf   --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct  --tasks triviaqa,truthfulqa_gen   --num_fewshot 0   --device cuda:2   --batch_size auto   --limit 0.02   --output_path ./eval_out/mistral_triviaqa_truthfulqa   --log_samples




lm_eval --model steer_hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,\
steer_layer=20,steer_lambda=0.8,steer_span=1,\
steer_vec_path=/home/youyang7/projects/fact-enhancement/artifacts/factual_dirs.json,trust_remote_code=True \
  --tasks lambada_openai \
  --device cuda:1 \
  --batch_size auto \
  --limit 10 \
  --output_path ./out/steer_demo --log_samples
