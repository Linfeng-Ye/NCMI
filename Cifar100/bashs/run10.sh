CUDA_VISIBLE_DEVICES=4 python NCMI_gcenter_r.py -net resnet18 -gpu -lr 0.1 -feature_Temp 0.1 -wd 5e-4 -centmomentum 0.9 -centlr 1.0 >> test7.txt
CUDA_VISIBLE_DEVICES=4 python NCMI_gcenter_r.py -net resnet18 -gpu -lr 0.1 -feature_Temp 0.2 -wd 5e-4 -centmomentum 0.9 -centlr 1.0 >> test7.txt
CUDA_VISIBLE_DEVICES=4 python NCMI_gcenter_r.py -net resnet18 -gpu -lr 0.1 -feature_Temp 0.5 -wd 5e-4 -centmomentum 0.9 -centlr 1.0 >> test7.txt
CUDA_VISIBLE_DEVICES=4 python NCMI_gcenter_r.py -net resnet18 -gpu -lr 0.1 -feature_Temp 0.7 -wd 5e-4 -centmomentum 0.9 -centlr 1.0 >> test7.txt
CUDA_VISIBLE_DEVICES=4 python NCMI_gcenter_r.py -net resnet18 -gpu -lr 0.1 -feature_Temp 1.0 -wd 5e-4 -centmomentum 0.9 -centlr 1.0 >> test7.txt
CUDA_VISIBLE_DEVICES=4 python NCMI_gcenter_r.py -net resnet18 -gpu -lr 0.1 -feature_Temp 2.0 -wd 5e-4 -centmomentum 0.9 -centlr 1.0 >> test7.txt