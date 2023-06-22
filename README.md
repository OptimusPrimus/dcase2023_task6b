# Advancing Natural-Language Based Audio Retrieval with PaSST and Large Audio-Caption Data Sets


### Setup

#### Create environment
```shell
conda env create -f environment.yml
conda activate salsa
pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.24#egg=hear21passt'
CFLAGS='-O3 -march=native' pip install https://github.com/f0k/minimp3py/archive/master.zip
```

#### Setup data environment

The default data directory is `~/shared` ( can be changed via the `directories.data_dir` flag).
The data sets in this folder should follow this structure:
- clotho_v2
  - clotho_captions_{development,evaluation,validation}.csv
  - clotho_metadata_{development,evaluation,validation}.csv
  - {development,evaluation,validation} folders with audio files
- audioset
  - download custom [repository](git@gitlab.cp.jku.at:audio_datasets/audioset.git)
  - download the audiocaps files into the corresponding directory
- audiocaps 
  - download the [GitHub repository](https://github.com/cdjkim/audiocaps)
- wavecaps
  - [json_files](https://github.com/XinhaoMei/WavCaps/tree/master/data/json_files)
  - download audio files from [hugging face](https://huggingface.co/datasets/cvssp/WavCaps/tree/main/Zip_files) info corresponding folders
- tmp
  - this is where the compressed audio data sets will go
- audio_retrieval, (project name)
  - this is for model checkpoints, etc.
- clotho_gpt
  - copy from this repository


### Minimal Example

```shell
cd src
conda activate salsa

python -m experiments.audio_retrieval.train with data_loader.batch_size=64 data_loader.batch_size_eval=32 audio_loader.max_audio_length=30 audio_features.segment_length=10 audio_features.name=passt sentence_features.model=bert-base-uncased initial_tau=0.01 s_patchout_f=2 s_patchout_t=15 lr=2e-5 min_lr=1e-7 rampdown_type=cosine max_epochs=16 rampdown_stop=15 warmup_length=1 rampdown_start=1 audio_features.adopt_n_layers=0 sentence_features.adopt_n_layers=0 train_on=clothov2 load_parameters=None gpt_augment_p=0.0
```

More examples can be found in the script folder.

