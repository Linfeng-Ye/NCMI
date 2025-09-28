CUDA_VISIBLE_DEVICES=3 python NCMI_gcenter.py -net resnet101 -gpu -lr 0.1 -feature_Temp 0.2 -wd 0.0005 \
                                                -centmomentum 0.999999 -cdecayfactor 0.999999 -centlr 1.0 >> res101.txt
CUDA_VISIBLE_DEVICES=3 python NCMI_gcenter.py -net resnet101 -gpu -lr 0.1 -feature_Temp 0.3 -wd 0.0005 \
                                                -centmomentum 0.999999 -cdecayfactor 0.999999 -centlr 1.0 >> res101.txt
CUDA_VISIBLE_DEVICES=3 python NCMI_gcenter.py -net resnet101 -gpu -lr 0.1 -feature_Temp 0.5 -wd 0.0005 \
                                                -centmomentum 0.999999 -cdecayfactor 0.999999 -centlr 1.0 >> res101.txt
