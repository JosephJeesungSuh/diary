python scripts/run_identity_survey.py \
   question_filepath="diary/data/questions/identity_survey/identity_is_smoker.json" \
   agent_params.agent_filepath="/experiment_data/diary_bank/narrative_gen/PROVIDE_FILENAME.jsonl" \
   interview_params.n_parallel=32 \
   sampling_params.model_name="mistralai/Mistral-Small-24B-Instruct-2501" \
   sampling_params.port=8000 \
   sampling_params.min_p=0.02 \
   save_dir=/experiment_data \
   hydra.run.dir='/experiment_data/hydra_logs/${now:%Y-%m-%d}/${now:%H-%M-%S}'