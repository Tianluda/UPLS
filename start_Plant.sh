cd ./UPLS

tags="CCAM_Apple_Plant"
train_options="10"
test_options="2"
cbam="False"
NewDisentanglers="2"
constraint_terms="False"
Disentangle_spatial="False"
Disentangle_cbam="True"
Disentangle_Fca="False"
models="0 1 2 3 4 9 best"

lr="0.0001"
batch_size="256"
pretrained="plant"
alpha="0.25"
max_epoch="10"
scales="2,3"
threshold="0.5"
crf_iteration="10"

tags_array=($tags)
train_options_array=($train_options)
test_options_array=($test_options)
cbam_array=($cbam)
NewDisentanglers_array=($NewDisentanglers)
constraint_terms_array=($constraint_terms)
Disentangle_spatial_array=($Disentangle_spatial)
Disentangle_cbam_array=($Disentangle_cbam)
Disentangle_Fca_array=($Disentangle_Fca)
length=${#tags_array[@]}

for ((i=0; i<length; i++)); do
    tag=${tags_array[i]}
    train_option=${train_options_array[i]}
    test_option=${test_options_array[i]}
    cbam_option=${cbam_array[i]}
    NewDisentangler=${NewDisentanglers_array[i]}
    constraint_term=${constraint_terms_array[i]}
    Disentangle_spatial=${Disentangle_spatial_array[i]}
    Disentangle_cbam=${Disentangle_cbam_array[i]}
    Disentangle_Fca=${Disentangle_Fca_array[i]}
    echo "Running train for tag=${tag}, train_option: ${train_option}, CBAM: ${cbam_option}, NewDisentangler=${NewDisentangler}, constraint_term=${constraint_term}, Disentangle_spatial=${Disentangle_spatial}, Disentangle_cbam=${Disentangle_cbam}, Disentangle_Fca=${Disentangle_Fca}"
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=3,4 /home/omnisky/python/envs/Tian_env/bin/python train_CCAM.py --tag $tag --batch_size $batch_size --pretrained $pretrained --alpha $alpha --max_epoch $max_epoch --CBAM $cbam_option --NewDisentangler $NewDisentangler --constraint_term $constraint_term --option $train_option --lr $lr --Disentangle_spatial $Disentangle_spatial --Disentangle_cbam $Disentangle_cbam --Disentangle_Fca $Disentangle_Fca
    echo "Ended train for tag=${tag}, train_option: ${train_option}, CBAM: ${cbam_option}, NewDisentangler=${NewDisentangler}, constraint_term=${constraint_term}, Disentangle_spatial=${Disentangle_spatial}, Disentangle_cbam=${Disentangle_cbam}, Disentangle_Fca=${Disentangle_Fca}"
done
for ((i=0; i<length; i++)); do
    tag=${tags_array[i]}
    train_option=${train_options_array[i]}
    test_option=${test_options_array[i]}
    cbam_option=${cbam_array[i]}
    NewDisentangler=${NewDisentanglers_array[i]}
    constraint_term=${constraint_terms_array[i]}
    Disentangle_spatial=${Disentangle_spatial_array[i]}
    Disentangle_cbam=${Disentangle_cbam_array[i]}
    Disentangle_Fca=${Disentangle_Fca_array[i]}
    for model_num in $models; do
        echo "Running inference for tag=${tag}, test_option: ${test_option}, CBAM: ${cbam_option}, NewDisentangler=${NewDisentangler}, Disentangle_spatial=${Disentangle_spatial}, Disentangle_cbam=${Disentangle_cbam}, Disentangle_Fca=${Disentangle_Fca}, scales=${scales}, threshold=${threshold},crf_iteration=${crf_iteration}, model_num=${model_num}"
        OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=3 /home/omnisky/python/envs/Tian_env/bin/python inference.py --tag $tag --scales $scales --option $test_option --CBAM $cbam_option --NewDisentangler $NewDisentangler --Disentangle_spatial $Disentangle_spatial --Disentangle_cbam $Disentangle_cbam --Disentangle_Fca $Disentangle_Fca --model_num $model_num --threshold $threshold --crf_iteration $crf_iteration
        echo "Ended inference for tag=${tag}, test_option: ${test_option}, CBAM: ${cbam_option}, NewDisentangler=${NewDisentangler}, Disentangle_spatial=${Disentangle_spatial}, Disentangle_cbam=${Disentangle_cbam}, Disentangle_Fca=${Disentangle_Fca}, scales=${scales}, threshold=${threshold},crf_iteration=${crf_iteration}, model_num=${model_num}"
    done
done

# ==========================================================================================

# 如果你想要将输出保存到日志文件中，你可以取消注释以下行
# nohup /bin/bash ./UPLS/start_Plant.sh >> ./UPLS/cron_Plant.log 2>&1