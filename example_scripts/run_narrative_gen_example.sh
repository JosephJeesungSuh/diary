python scripts/run_narrative_gen.py \
   question_filepath="diary/data/questions/narrative_generation/interview_smoking.json" \
   agent_params.n_agent=1 \
   interview_params.n_parallel=1 \
   sampling_params.model_name="meta-llama/Llama-3.1-8B" \
   sampling_params.port=8000 \
   sampling_params.min_p=0.01 \
   save_dir=/experiment_data \
   hydra.run.dir=/experiment_data/hydra_logs/${now:%Y-%m-%d}/${now:%H-%M-%S}