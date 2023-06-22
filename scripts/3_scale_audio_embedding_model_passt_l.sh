#!/bin/bash
#
#  usage: sbatch ./gpu_test.scrpt
#
#SBATCH -J salsa
#SBATCH --partition zen2_0256_a40x2
#SBATCH --qos zen2_0256_a40x2            #zen3_0512_a100x2
#SBATCH --gres=gpu:1                     #or --gres=gpu:1 if you only want to use half a node
#SBATCH -o /home/fs71983/primusp/slurm/TRA-%x.%j_%a.out
# #SBATCH --array=0-2
#SBATCH --ntasks-per-node=16

source $HOME/.bashrc


export MEM_FOLDER=/tmp/kk_datasets

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

N_ADAPT_LAYERS=0
SEGMENT_LENGTH=10
AUDIO_FEATURES=passt_l
SENTENCE_FEATURES=bert-base-uncased
EDA=0.0
EDA_SYN=0.3
EDA_DEL=0.2
TRANSLATE_AUGMENT=0.0
GPT_AUGMENT=0.0

PATCHOUT_F=3
PATCHOUT_T=50

TAU=0.01
TRAIN_ON=allplus
LOAD_PARAMETERS=None

MAX_EPOCHS=16
MIN_LR=1e-7
LR=2e-5
RD_START=1
RD_STOP=15

conda activate salsa
cd $HOME/salsa/src

OMP_NUM_THREADS=1 python -m experiments.audio_retrieval.train with data_loader.batch_size=64 data_loader.batch_size_eval=32 audio_loader.max_audio_length=30 audio_features.segment_length=$SEGMENT_LENGTH audio_features.name=$AUDIO_FEATURES sentence_features.model=$SENTENCE_FEATURES eda_p=$EDA eda_p_syn=$EDA_SYN eda_p_del=$EDA_DEL translate_augment_p=$TRANSLATE_AUGMENT initial_tau=$TAU s_patchout_f=$PATCHOUT_F s_patchout_t=$PATCHOUT_T lr=$LR min_lr=$MIN_LR rampdown_type=cosine max_epochs=$MAX_EPOCHS rampdown_stop=$RD_STOP warmup_length=$RD_START rampdown_start=$RD_START audio_features.adopt_n_layers=$N_ADAPT_LAYERS sentence_features.adopt_n_layers=$N_ADAPT_LAYERS train_on=$TRAIN_ON load_parameters=$LOAD_PARAMETERS gpt_augment_p=$GPT_AUGMENT

rm -rf $MEM_FOLDER
