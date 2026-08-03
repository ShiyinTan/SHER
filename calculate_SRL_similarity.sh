## 分别create coref和srl后，再在coref.ipynb中将两者合并，并且删去多余字符

# 设置多个devices：0,1
if [ "$#" -ge 1 ]; then
    gpu="$1"
else
    gpu=0
fi

if [ "$#" -ge 2 ]; then
    data_name="$2"
else
    data_name=0
fi


output_file="calculate_SRL_similarity_${data_name}.out"

host_name=$(hostname)
current_time=$(date)

echo "host_name: $host_name in time: $current_time" > $output_file


CUDA_VISIBLE_DEVICES=${gpu} nohup python -u calculate_SRL_similarity.py --data_name ${data_name} >> $output_file 2>&1 &



# CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main-modify.py --mode ${mode} > primer-${mode}_${par_id}.out 2>&1 &