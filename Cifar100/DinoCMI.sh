#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00-12:00
#SBATCH --output=out/Woof_IPC10%N-%j.txt
#SBATCH --job-name=Woof_IPC10
#SBATCH --account=def-ehyang-it

module load python/3.10 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install -q --no-index --upgrade pip
pip install -q --no-index -r requirements.txt

python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 0.1 -feature_Temp 0.07 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 0.1 -feature_Temp 0.1 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 0.1 -feature_Temp 0.2 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 0.1 -feature_Temp 0.5 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 0.1 -feature_Temp 1.0 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 0.1 -feature_Temp 2.0 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 1.0 -feature_Temp 0.07 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 1.0 -feature_Temp 0.1 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 1.0 -feature_Temp 0.2 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 1.0 -feature_Temp 0.5 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 1.0 -feature_Temp 1.0 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 1.0 -feature_Temp 2.0 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 2.0 -feature_Temp 0.07 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 2.0 -feature_Temp 0.1 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 2.0 -feature_Temp 0.2 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 2.0 -feature_Temp 0.5 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 2.0 -feature_Temp 1.0 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 2.0 -feature_Temp 2.0 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 5.0 -feature_Temp 0.07 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 5.0 -feature_Temp 0.1 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 5.0 -feature_Temp 0.2 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 5.0 -feature_Temp 0.5 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 5.0 -feature_Temp 1.0 >> test.txt 
python CMI_Sharp_train.py -net resnet18 -gpu -warm 5 -lr 5.0 -feature_Temp 2.0 >> test.txt 
