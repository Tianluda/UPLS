cd ./segment-anything-main/notebooks

CUDA_VISIBLE_DEVICES=0 /home/omnisky/python/envs/sam/bin/python C2AM_SAM_Box.py --dir_a /media/omnisky/tld/TianLuDa/CCAM/CUSTOM_new/分类评估/CCAM_Apple_DETCO@scale=2,3@t=0.5@ccam_inference_crf=10/ --dir_b /media/omnisky/tld/Tian_Datasets/万张图/Resized_Leaf_Foreground/ --save_path /media/omnisky/tld/TianLuDa/CCAM/CUSTOM_new/分类评估/CCAM_Apple_DETCO_SAM_Box200_Point_ratio5
CUDA_VISIBLE_DEVICES=0 /home/omnisky/python/envs/sam/bin/python C2AM_SAM_Box.py --dir_a /media/omnisky/tld/TianLuDa/CCAM/CUSTOM_new/分类评估/CCAM_Apple_Plant@scale=2,3@t=0.5@ccam_inference_crf=10/ --dir_b /media/omnisky/tld/Tian_Datasets/万张图/Resized_Leaf_Foreground/ --save_path /media/omnisky/tld/TianLuDa/CCAM/CUSTOM_new/分类评估/CCAM_Apple_Plant_SAM_Box200_Point_ratio5

CUDA_VISIBLE_DEVICES=0 /home/omnisky/python/envs/sam/bin/python C2AM_SAM_Box.py --dir_a /media/omnisky/tld/TianLuDa/CCAM/CUSTOM_new/分类评估/CCAM_Extra_DETCO@scale=2,3@t=0.5@ccam_inference_crf=10/ --dir_b /media/omnisky/tld/Tian_Datasets/万张图/Extra_Original_Resized_Leaf/Original_picture/ --save_path /media/omnisky/tld/TianLuDa/CCAM/CUSTOM_new/分类评估/CCAM_Extra_DETCO_SAM_Box200_Point_ratio5
CUDA_VISIBLE_DEVICES=0 /home/omnisky/python/envs/sam/bin/python C2AM_SAM_Box.py --dir_a /media/omnisky/tld/TianLuDa/CCAM/CUSTOM_new/分类评估/CCAM_Extra_Plant@scale=2,3@t=0.5@ccam_inference_crf=10/ --dir_b /media/omnisky/tld/Tian_Datasets/万张图/Extra_Original_Resized_Leaf/Original_picture/ --save_path /media/omnisky/tld/TianLuDa/CCAM/CUSTOM_new/分类评估/CCAM_Extra_Plant_SAM_Box200_Point_ratio5

# 如果你想要将输出保存到日志文件中，你可以取消注释以下行
# nohup ./segment-anything-main/notebooks/C2AM_SAM_Box.sh >> ./segment-anything-main/notebooks/C2AM_SAM_Box.log 2>&1