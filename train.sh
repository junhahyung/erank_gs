CUDA_VISIBLE_DEVICES=0 python train.py -s ../DTU/scan105 --exp_name "erankgs_scan105" --iterations 30000 --erank_lambda 0.01 --erank_from_iter 7000 --reg_from_iter 35000  --thin_lambda 1  -r 2 
