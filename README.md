#Example RUN:
python3 source/main.py --dataset billsum --fp16 --num_epochs 20 --dataset_cleaned --chunk_min_len 256 --chunk_max_len 1024 --n_instances_eval 10 --patience 5 --min_length 100 --max_output_size 300 --n_instances_train 10