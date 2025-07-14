python scripts/run_treatment.py \
    treatment=control \
    treatment.environment_filepath="diary/data/questions/treatment/control_daily.json" \
    treatment.agent_params.agent_filepath=diary_bank/narrative_gen/00001_interview_smoking.jsonl \
    treatment.sampling_params.model_name="meta-llama/Llama-3.1-8B-Instruct" \
    treatment.sampling_params.port=8001 \
    treatment.interview_params.n_parallel=1