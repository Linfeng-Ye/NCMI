CUDA_VISIBLE_DEVICES=0 python NCMI_gcenter.py -net resnet34 -gpu -lr 0.1 -feature_Temp 0.2 -wd 0.0005 \
                                                -centmomentum 0.999999 -cdecayfactor 0.999999 -centlr 5.0 >> Res34.txt
CUDA_VISIBLE_DEVICES=0 python NCMI_gcenter.py -net resnet34 -gpu -lr 0.1 -feature_Temp 0.3 -wd 0.0005 \
                                                -centmomentum 0.999999 -cdecayfactor 0.999999 -centlr 5.0 >> Res34.txt
CUDA_VISIBLE_DEVICES=0 python NCMI_gcenter.py -net resnet34 -gpu -lr 0.1 -feature_Temp 0.5 -wd 0.0005 \
                                                -centmomentum 0.999999 -cdecayfactor 0.999999 -centlr 5.0 >> Res34.txt
