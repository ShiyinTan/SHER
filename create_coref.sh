# 设置多个devices：0,1
if [ "$#" -ge 1 ]; then
    gpu="$1"
else
    gpu=0
fi



output_file="create_coref2.out"

host_name=$(hostname)
current_time=$(date)

echo "host_name: $host_name in time: $current_time" > $output_file


CUDA_VISIBLE_DEVICES=${gpu} nohup python -u create_coref.py >> $output_file 2>&1 &



# CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main-modify.py --mode ${mode} > primer-${mode}_${par_id}.out 2>&1 &