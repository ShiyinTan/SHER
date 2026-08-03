
# 设置多个devices：0,1
if [ "$#" -ge 1 ]; then
    gpu="$1"
else
    gpu=0
fi

if [ "$#" -ge 2 ]; then
    par_str="$2"
else
    par_str="default"
fi


output_file="calculate_SRLscore_${par_str}.out"

host_name=$(hostname)
current_time=$(date)

echo "host_name: $host_name in time: $current_time" > $output_file


CUDA_VISIBLE_DEVICES=${gpu} nohup python calculate_SRLscore_test.py >> $output_file 2>&1 &



# CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main-modify.py --mode ${mode} > primer-${mode}_${par_id}.out 2>&1 &

