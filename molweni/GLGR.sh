
CUDA_VISIBLE_DEVICES=0 python GLGR.py \
		--model_type electra \
		--model_name_or_path /SISDC_GPFS/Home_SE/hy-suda/pre-train_model/electra-large/ \
		--do_train \
		--do_eval \
		--data_dir /SISDC_GPFS/Home_SE/hy-suda/lyl/graph/final_data/molweni \
		--train_file train \
		--dev_file dev \
		--test_file test \
		--output_dir /SISDC_GPFS/Home_SE/hy-suda/lyl/MDRC-Graph/molweni/output/electra-g1g2 \
		--overwrite_output_dir \
		--per_gpu_train_batch_size 12 \
		--gradient_accumulation_steps 1 \
		--per_gpu_eval_batch_size 8 \
		--num_train_epochs 3 \
		--learning_rate 1e-5 \
		--threads 20 \
		--do_lower_case \
		--evaluate_during_training \
		--max_answer_length 30 \
		--logging_steps 500 \
		--weight_decay 0.01 \
		--max_seq_length 384 \
		--seed 15 \
		--fp16 --fp16_opt_level "O2" 
done

