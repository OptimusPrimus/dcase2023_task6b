import os
from abc import ABC
import string
import math
import sys
import pytorch_lightning as pl
import wandb
import copy
import itertools

from torch.utils.data import ConcatDataset


from data.datasets.clotho_v2 import clotho_v2, get_clotho_v2
from data.datasets.audio_caps import audiocaps, get_audiocaps
from data.datasets.wavecaps import wavecaps, get_wavecaps
from data.datasets.gpt_augment import gpt_augment, GPTAugment

from data.data_loader import data_loader, get_train_data_loader, get_eval_data_loader
from data.audio_loader import audio_loader, load_cached_audio_set

from sacred import Experiment

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import AutoModel, AutoTokenizer
import torch

from glob import glob
from pytorch_lightning.loggers import WandbLogger
from utils.directories import directories, get_model_dir
import numpy as np
import torch.distributed as dist


wandb.login()
audio_retrieval = Experiment('audio_retrieval', ingredients=[
    directories,
    clotho_v2,
    audiocaps,
    wavecaps,
    data_loader,
    audio_loader,
    gpt_augment
])

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@audio_retrieval.config
def default_config():

    log_db_name = 'audio_retrieval_workshop'

    # model loading and saving
    load_parameters = None
    load_last = 'best'
    resume_training = None

    audio_features = {
        'name': 'cnn10',
        'type': 'embedding',
        'frozen': False,
        'adopt_n_layers': 0,
        'adopt_layer_size': 2048,
        'segment_length': 30,
        'aggregate': 'mean'
    }

    sentence_features = {
        'model': 'bert-base-uncased',
        'frozen': False,
        'adopt_n_layers': 0,
        'adopt_layer_size': 2048,
        'use_eos_token_as_embedding': True,
        'max_sentence_tokens': 32
    }

    # loss function
    initial_tau = 0.02
    shared_representation_size = 1024
    normalize = True

    # data set & augmentations
    train_on = 'clothov2'
    eval_on_test = False

    # audio augmentations
    s_patchout_t = 0
    s_patchout_f = 0

    time_drop_width = 0
    time_stripes_num = 0
    freq_drop_width = 0
    freq_stripes_num = 0

    # text augmentations
    gpt_augment_p = 0.0

    translate_augment_p = 0
    translate_augment_languages = ['de', 'es', 'fr'] # "tr"

    eda_p = 0.0
    eda_p_swap = 0.0
    eda_p_ins = 0.0
    eda_p_del = 0.0
    eda_p_syn = 0.0

    # optimizer
    max_epochs = 25
    max_samples_per_epoch = None
    adamw = True
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.0
    amsgrad = False
    lr = 2e-5
    min_lr = 1e-7
    hard_steps = False
    warmup_length = 3
    rampdown_start = 3
    rampdown_stop = 25
    rampdown_type = 'cosine'

    eps = 1e-8
    accumulate_grad_batches = 1
    gradient_clip_val = None

    # technical stuff
    gpus = 1
    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    half_precision = True
    fast_dev_run = False
    strategy = None
    monitor = 'mAP@10'
    enable_checkpointing = True

    save_ram = False # store data set file in /dev/shm to save memory on nodes with multiple experiments

    num_nodes=1


@audio_retrieval.capture
def run_one(log_db_name, train_on, resume_training, eval_on_test, monitor, translate_augment_languages, gpt_augment_p, _config=None):

    # make un-mutable dict mutable
    _config = dict(_config)

    print('Initialize model...')
    model = get_model()

    # create the data module
    print(f'Loading {train_on} data sets...')
    train = get_data_set(train_on.split('_')[0], 'train') if not _config['fast_dev_run'] else get_data_set(train_on.split('_')[0], 'test')
    val = get_data_set('clothov2', 'val') if not _config['fast_dev_run'] else get_data_set('clothov2', 'test')
    test = get_data_set('clothov2', 'test') if not _config['fast_dev_run'] else get_data_set('clothov2', 'test')
    predict = get_data_set('clothov2', 'predict') if not _config['fast_dev_run'] else get_data_set('clothov2', 'predict')

    if train_on.split('_')[0] == 'clothov2' and gpt_augment_p > 0:
        print('Using GPT augment.')
        assert train_on.split('_')[0] == 'clothov2', f'GPT augmentation not supported for {train} yet.'
        train = GPTAugment(train, gpt_augment_p)

    print(f'Training set size: {len(train)}')
    print(f'Val set size: {len(val)}')
    print(f'Test set size: {len(test)}')

    # training
    print('Start training...')
    distributed = _config['num_nodes'] > 1
    val_set = get_eval_data_loader(val, collate_fun=collate_fun, shuffle=True, distributed=distributed)
    if distributed and int(os.environ['NODE_RANK']) > 0:
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(log_model=False, project='dev' if _config['fast_dev_run'] else log_db_name)
    t = get_trainer(wandb_logger)
    t.fit(
        model,
        train_dataloaders=get_train_data_loader(train, collate_fun=collate_fun),
        val_dataloaders=val_set,
        ckpt_path=os.path.join(get_model_dir(), resume_training, 'last.ckpt') if resume_training else None
    )

    print('Start testing on last...')
    test_result = t.test(model, get_eval_data_loader(test, collate_fun=collate_fun, shuffle=True, distributed=False))

    return test_result[0][f'test/{monitor}']

def collate_fun(individual_samples):

    batch = {
        'audio': torch.stack([torch.from_numpy(s['audio']) for s in individual_samples], dim=0),
        'audio_length': torch.tensor([s['audio_length'] for s in individual_samples]),
        'caption': [s['caption'] for s in individual_samples],
        'path': [s['path'] for s in individual_samples],
        'idx': torch.tensor([s['idx'] for s in individual_samples])
    }
    return batch


@audio_retrieval.capture
def get_data_set(data_set_id, mode, save_ram, _config):

    # load training files and audio WAVs if necesary
    if data_set_id == 'clothov2':
        assert mode in ['train', 'val', 'evl', 'test', 'predict', 'analysis']
        ds = get_clotho_v2(mode)
        ds = load_cached_audio_set(ds, compress=False, shared=save_ram)
        return ds
    elif data_set_id == 'wavecaps':
        ds = load_cached_audio_set(get_wavecaps(), compress=True, shared=save_ram)
        return ds
    elif data_set_id == 'clothov2+wavecaps':
        assert mode in ['train']
        ds = [
            load_cached_audio_set(get_wavecaps(), compress=True, shared=save_ram),
            load_cached_audio_set(get_clotho_v2('train'), compress=False, shared=save_ram)
        ]
        return torch.utils.data.ConcatDataset(ds)
    elif data_set_id == 'clothov2+audiocaps':
        assert mode in ['train']
        ds = [load_cached_audio_set(get_audiocaps(mode), compress=True, shared=save_ram) for mode in
              ['train', 'val', 'test']]
        ds.append(load_cached_audio_set(get_clotho_v2('train'), compress=False, shared=save_ram))
        return torch.utils.data.ConcatDataset(ds)
    elif data_set_id == 'wavecaps+audiocaps':
        assert mode in ['train']
        ds = [load_cached_audio_set(get_audiocaps(mode), compress=True, shared=save_ram) for mode in
              ['train', 'val', 'test']]
        ds.append(load_cached_audio_set(get_wavecaps(), compress=True, shared=save_ram))
        return torch.utils.data.ConcatDataset(ds)
    elif data_set_id == 'audiocaps':
        assert mode in ['train']
        # if audio_caps, load all of it...
        ds = [load_cached_audio_set(get_audiocaps(mode), compress=True, shared=save_ram) for mode in ['train', 'val', 'test']]
        return torch.utils.data.ConcatDataset(ds)
    elif data_set_id == 'all':
        ds = [
                 load_cached_audio_set(get_wavecaps(), compress=True, shared=save_ram),
                 load_cached_audio_set(get_clotho_v2('train'), compress=False, shared=save_ram),
             ] + [load_cached_audio_set(get_audiocaps(mode), compress=True, shared=save_ram) for mode in ['train', 'val', 'test']]
        return torch.utils.data.ConcatDataset(ds)

    else:
        raise NotImplementedError(f'Data set {data_set_id} unknown.')


@audio_retrieval.capture
def get_model(load_parameters, _config):
    ac = AudioRetrievalModel(**_config)

    # init parameters from pre-trained model
    if load_parameters:
        print(f'Loading model {load_parameters} ...')
        save_dir = os.path.join(os.path.expanduser('~'), 'shared', 'audioset_tagging', 'model_checkpoints', load_parameters)
        assert os.path.exists(save_dir)
        if _config['load_last'] == 'last':
            print('Loading last checkpoint.')
            model_path = list(glob(os.path.join(save_dir, 'last.ckpt')))[-1]
        elif _config['load_last'] == 'best':
            print('Loading best checkpoint.')
            paths = glob(os.path.join(save_dir, 'epoch_*.ckpt'))
            paths.sort(key=lambda x: float(os.path.basename(x).split('-')[-1].split('.')[1]))
            model_path = paths[-1]
        else:
            raise AttributeError(_config['load_last'])
        print(model_path)
        ac_ = AudioRetrievalModel.load_from_checkpoint(model_path)
        missing_keys = ac.load_state_dict(ac_.state_dict())
        print(missing_keys)

    return ac


@audio_retrieval.capture
def get_audio_embedding_model(_config, s_patchout_t=0, s_patchout_f=0,
                        time_drop_width=1, time_stripes_num=0,
                        freq_drop_width=1, freq_stripes_num=0):

    if _config['audio_features']['name'].startswith('passt'):

        from hear21passt.base import get_basic_model, get_model_passt
        from hear21passt.models.preprocess import AugmentMelSTFT
        import torch
        # get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
        if "passt_s" == _config['audio_features']['name']: 
            print("#### Using PaSST-S ap486 model with no overlap ####\n")
            model = get_model_passt("passt_s_kd_p16_128_ap486", input_tdim=998, fstride=10, tstride=10, s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)
        elif "passt_20" == _config['audio_features']['name']:
            print("#### Using PaSST-S  train on `20` seconds ####\n")
            model = get_model_passt(arch="passt_20sec", input_tdim=2000, fstride=10, tstride=10, s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)
        elif "passt_l" == _config['audio_features']['name']:
            print("#### Using PaSST-L  ####\n")
            model = get_model_passt(arch="passt_l_kd_p16_128_ap47", input_tdim=998, fstride=10, tstride=10, s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)
        else:
            print("#### Using PaSST model with no overlap ####")
            model = get_model_passt("passt_s_p16_s16_128_ap468", input_tdim=1000, fstride=16, tstride=16, s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)

        model.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                                   timem=192,
                                   htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                                   fmax_aug_range=2000)

        audio_embedding_model = model

        class Wrapper(torch.nn.Module):

            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x, **kwargs):
                with torch.no_grad():
                    mel = audio_embedding_model.mel(x)
                return audio_embedding_model(mel[:, None])[-1]

        return Wrapper(audio_embedding_model), 768

    elif _config['audio_features']['name'].startswith('cnn'):
        import torch
        from architecture.panns_original import Cnn10, Cnn14
        if _config['audio_features']['name'] == 'cnn10':
            print('Using CNN10')
            cnn = Cnn10(32000, 1024, 320, 64, 50, 14000, 527,
                        time_drop_width=time_drop_width, time_stripes_num=time_stripes_num,
                        freq_drop_width=freq_drop_width, freq_stripes_num=freq_stripes_num)
            state = torch.load(os.path.join(os.path.expanduser('~'), 'shared/panns/Cnn10_mAP=0.380.pth'), map_location=torch.device('cpu'))
            cnn.load_state_dict(state['model'])
            embedding_size = 512
        else:
            print('Using CNN14')
            cnn = Cnn14(32000, 1024, 320, 64, 50, 14000, 527,
                        time_drop_width=time_drop_width, time_stripes_num=time_stripes_num,
                        freq_drop_width=freq_drop_width, freq_stripes_num=freq_stripes_num)
            state = torch.load(os.path.join(os.path.expanduser('~'), 'shared/panns/Cnn14_mAP=0.431.pth'), map_location=torch.device('cpu'))
            cnn.load_state_dict(state['model'])
            embedding_size = 2048

        class Wrapper(torch.nn.Module):

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x, lengths=None, **kwargs):
                return self.model(x, lengths=lengths)['embedding']

        return Wrapper(cnn), embedding_size




@audio_retrieval.capture
def get_sentence_embedding_model(_config):

    from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer, \
        RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer, \
        CLIPTokenizer, CLIPTextModel

    MODELS = {
        'openai/clip-vit-base-patch32': (CLIPTextModel, CLIPTokenizer, 512),
        'prajjwal1/bert-tiny': (BertModel, BertTokenizer, 128),
        'prajjwal1/bert-mini': (BertModel, BertTokenizer, 256),
        'prajjwal1/bert-small': (BertModel, BertTokenizer, 512),
        'prajjwal1/bert-medium': (BertModel, BertTokenizer, 512),
        'gpt2': (GPT2Model, GPT2Tokenizer, 768),
        'distilgpt2': (GPT2Model, GPT2Tokenizer, 768),
        'bert-base-uncased': (BertModel, BertTokenizer, 768),
        'bert-large-uncased': (BertModel, BertTokenizer, 1024),
        'roberta-base': (RobertaModel, RobertaTokenizer, 768),
        'roberta-large': (RobertaModel, RobertaTokenizer, 1024),
        'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768),
        "distilroberta-base": (RobertaModel, RobertaTokenizer, 768),
        "sentence-transformers/clip-ViT-B-32-multilingual-v1": (AutoModel, AutoTokenizer, 768)
    }

    if 'clip' not in _config['sentence_features']['model']:
        sentence_embedding_model = MODELS[_config['sentence_features']['model']][0].from_pretrained(_config['sentence_features']['model'],
                                                                 add_pooling_layer=False,
                                                                 hidden_dropout_prob=0.2,
                                                                 attention_probs_dropout_prob=0.2,
                                                                 output_hidden_states=False)
    else:
        sentence_embedding_model = MODELS[_config['sentence_features']['model']][0].from_pretrained(_config['sentence_features']['model'])

    tokenizer = MODELS[_config['sentence_features']['model']][1].from_pretrained(_config['sentence_features']['model'])

    return sentence_embedding_model, tokenizer, MODELS[_config['sentence_features']['model']][2]


class AudioRetrievalModel(pl.LightningModule, ABC):

    def __init__(
            self,
            **kwargs
    ):

        super().__init__()
        self.save_hyperparameters(kwargs)

        self.kwargs = kwargs

        self.distributed_mode = kwargs.get('num_nodes', 1) > 1

        self.audio_embedding_model, audio_output_size = get_audio_embedding_model()

        self.sentence_embedding_model, self.tokenizer, text_output_size = get_sentence_embedding_model()

        layer_sizes = [audio_output_size]
        layer_sizes += [self.kwargs['audio_features']['adopt_layer_size']] * self.kwargs['audio_features']['adopt_n_layers']
        layer_sizes += [self.kwargs['shared_representation_size']]
        audio_layers = []
        for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
            audio_layers.append(torch.nn.Linear(i, o))
            audio_layers.append(torch.nn.ReLU())

        audio_layers.pop()
        self.project_audio = torch.nn.Sequential(*audio_layers)

        layer_sizes = [text_output_size]
        layer_sizes += [self.kwargs['sentence_features']['adopt_layer_size']] * self.kwargs['sentence_features']['adopt_n_layers']
        layer_sizes += [self.kwargs['shared_representation_size']]
        sentence_layers = []
        for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
            sentence_layers.append(torch.nn.Linear(i, o))
            sentence_layers.append(torch.nn.ReLU())

        sentence_layers.pop()
        self.project_sentence = torch.nn.Sequential(*sentence_layers)

        self.tau = torch.nn.Parameter(torch.zeros((1,)) + self.kwargs['initial_tau'])
        self.loss = torch.nn.LogSoftmax(dim=1)


    def forward_audio(self, batch, audio_embedding_model, _config):
        # only compute audio features, if not pre-computed
        if 'audio_features' in batch:
            return batch

        # freeze audio embedding if required
        if _config['audio_features']['frozen']:
            audio_embedding_model = audio_embedding_model.eval()

        # embed audios
        with torch.set_grad_enabled(not _config['audio_features']['frozen']):
            # embed the whole audio sequence

            segment_length = _config['audio_features']['segment_length'] * 32000
            longest_audio = int((batch['audio'].shape[-1] * max(batch['audio_length'])).item())
            n_segments = int(math.ceil(longest_audio / segment_length))
            max_length = int(n_segments * segment_length)

            batch['audio_length'] = (batch['audio_length'] * batch['audio'].shape[-1]) / (segment_length * n_segments)
            batch['audio'] = batch['audio'][:, :max_length]

            split = torch.split(batch['audio'], segment_length, -1)
            S = len(split)
            B, L = split[0].shape
            split = torch.concatenate(split) # (B*S, L)
            embedding_sequence = torch.stack(torch.split(audio_embedding_model(split), B)).permute(1, 0, 2) # (B*S, L) -> (S, B, L) -> (B, S, L)
            embeddings = []

            if _config['audio_features']['aggregate'] == 'mean':
                for l, s in zip(batch['audio_length'], embedding_sequence):
                    l = math.ceil(S * l)
                    embeddings.append(s[:l].mean(0))
            else:
                raise ValueError

            embeddings = torch.stack(embeddings)
            batch['audio_features'] = embeddings

        return batch

    @staticmethod
    def forward_sentence(batch, sentence_embedding_model, tkz, _config):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        captions = []
        for i, b in enumerate(batch['caption']):
            if not (type(b) == str):
                print(b)
                b = b[0]
            captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))


        tokenized = tkz(
            captions,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt'
        )

        batch['input_ids'] = tokenized['input_ids'].to(device)
        batch['attention_mask'] = tokenized['attention_mask'].to(device)

        if tokenized['input_ids'].shape[1] > _config['sentence_features']['max_sentence_tokens']:
            batch['input_ids'] = batch['input_ids'][:, :_config['sentence_features']['max_sentence_tokens']]
            batch['attention_mask'] = batch['attention_mask'][:, :_config['sentence_features']['max_sentence_tokens']]

        # freeze audio embedding if required
        if _config['sentence_features']['frozen']:
            sentence_embedding_model = sentence_embedding_model.eval()

        # embed audios
        with torch.set_grad_enabled(not _config['sentence_features']['frozen']):
            token_embeddings = sentence_embedding_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[0]
            if _config['sentence_features']['use_eos_token_as_embedding']:
                batch['sentence_features'] = token_embeddings[:, 0, :]
            else:
                input_mask_expanded = batch['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                batch['sentence_features'] = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return batch

    def forward(self, batch):

        batch = self.forward_sentence(batch, self.sentence_embedding_model, self.tokenizer, self.kwargs)
        batch = self.forward_audio(batch, self.audio_embedding_model, self.kwargs)

        audio_features = self.project_audio(batch['audio_features'])
        sentence_features = self.project_sentence(batch['sentence_features'])

        if self.kwargs['normalize']:
            audio_features = torch.nn.functional.normalize(audio_features, p=2, dim=1)
            sentence_features = torch.nn.functional.normalize(sentence_features, p=2, dim=1)

        return audio_features, sentence_features

    def training_step(self, batch, batch_idx):
        self.update_scalars(batch_idx)

        audio_features, sentence_features = self(batch)
        paths = np.array([hash(p) for p in batch['path']])

        if self.distributed_mode:
            paths_all = self.all_gather(paths).reshape(-1)
        else:
            paths_all = torch.tensor(paths)

        I = (paths_all.unsqueeze(0) == paths_all.unsqueeze(1))

        if self.distributed_mode:
            audio_features = self.all_gather(audio_features, sync_grads=True).reshape(-1, audio_features.shape[-1])
            sentence_features = self.all_gather(sentence_features, sync_grads=True).reshape(-1, sentence_features.shape[-1])


        assert len(audio_features) == len(sentence_features), f"Captions: {len(batch['caption'])}, Audios: {len(batch['audio'])}, Audio Features Shape: {audio_features.shape} Sentence Features Shape: {sentence_features.shape}"

        C = torch.einsum('ij, kj -> ik', audio_features, sentence_features) / torch.abs(self.tau)

        C_audio = torch.log_softmax(C, dim=0)
        C_text = torch.log_softmax(C, dim=1)

        assert C_audio.shape[0] == C_audio.shape[1], f'Audio Features Shape: {audio_features.shape} Sentence Features Shape: {sentence_features.shape}'
        assert C_text.shape[0] == C_text.shape[1]

        loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())
        self.log("train/loss", loss, batch_size=len(audio_features), sync_dist=True)

        self.log('train/tau', torch.abs(self.tau), sync_dist=True)

        return loss



    def validation_step(self, batch, batch_idx, dl_index=0, mode='val'):

        if dl_index == 1 and mode == 'val':
            mode = 'test'
        elif dl_index == 2 and mode == 'val':
            mode = 'predict'
        with torch.no_grad():
            audio_features, sentence_features = self(batch)

        return {
            'audio_features': copy.deepcopy(audio_features.detach()),
            'sentence_features': copy.deepcopy(sentence_features.detach()),
            'caption': batch['caption'],
            'path': batch['path'],
            'idx': batch['idx'],
            'mode': mode
        }

    def validation_step_end(self, *args, **kwargs):
        if type(args) is not tuple:
            args = [args]
        import itertools
        import numpy as np
        mode = args[0]['mode']
        paths = list(itertools.chain(*[batch['path'] for batch in args]))
        captions = list(itertools.chain(*[batch['caption'] for batch in args]))
        I = torch.from_numpy((np.array(paths)[:, None] == np.array(paths)[None, :])).type(torch.bool)
        args = {k: torch.cat([batch[k] for batch in args], dim=0) for k in args[0] if
                k in ['audio_features', 'sentence_features', 'audio_mask', 'attention_mask', 'idx']}

        audio_features = args['audio_features']
        sentence_features = args['sentence_features']

        C = torch.einsum('ij, kj -> ik', audio_features, sentence_features) / torch.abs(self.tau)

        C_audio = torch.log_softmax(C, dim=0)
        C_text = torch.log_softmax(C, dim=1)

        assert C_audio.shape[0] == C_audio.shape[
            1], f'Audio Features Shape: {audio_features.shape} Sentence Features Shape: {sentence_features.shape}'
        assert C_text.shape[0] == C_text.shape[1]

        # mode = kwargs.get('mode', 'val')
        loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())
        self.log(f"{mode}/loss", loss.item(), batch_size=len(audio_features), add_dataloader_idx=False, sync_dist=True)
        args['path'] = paths
        args['caption'] = captions
        return args

    def validation_epoch_end(self, outputs, mode='val'):
        if len(outputs) == 0:
            return
        if type(outputs[0]) == list and len(outputs) == 3:
            self.validation_epoch_end(outputs[0], mode='val')
            self.validation_epoch_end(outputs[1], mode='test')
            self.validation_epoch_end(outputs[2], mode='predict')
        if type(outputs[0]) != dict and mode == 'predict':
            outputs = outputs[0]

        import numpy as np
        paths = [p for b in outputs for p in b['path']]
        captions = [p for b in outputs for p in b['caption']]
        idxs = [i.item() for b in outputs for i in b['idx']]


        audio_features = torch.cat([o['audio_features'] for o in outputs])
        sentence_features = torch.cat([o['sentence_features'] for o in outputs])

        if self.distributed_mode:
            lp = len(paths)
            all_paths= [None for _ in range(word_size)]
            dist.all_gather_object(all_paths, paths)
            paths = list(itertools.chain(*all_paths))

            all_paths= [None for _ in range(word_size)]
            dist.all_gather_object(all_paths, captions)
            captions = list(itertools.chain(*all_paths))

            all_paths= [None for _ in range(word_size)]
            dist.all_gather_object(all_paths, idxs)
            idxs = list(itertools.chain(*all_paths))

            all_audio_features  = self.all_gather(audio_features)
            audio_features = all_audio_features.reshape(-1, audio_features.shape[-1])

            all_sentence_features  = self.all_gather(sentence_features)
            sentence_features = all_sentence_features.reshape(-1, sentence_features.shape[-1])

        _, sorted = np.unique(idxs, return_index=True)

        audio_features = audio_features[sorted]
        sentence_features = sentence_features[sorted]
        paths = np.array(paths)[sorted]
        captions = np.array(captions)[sorted]

        from collections import Counter
        n_captions = Counter(paths)
        assert [v == n_captions[paths[0]] for k, v in n_captions.items()]
        n_captions = n_captions[paths[0]]

        C = torch.empty((len(sentence_features), len(audio_features) // n_captions))
        for i, sentence in enumerate(sentence_features):
            C[i, :] = (sentence.unsqueeze(0) * audio_features[::n_captions]).mean(-1)

        if self.trainer.is_global_zero:
            experiment_name = 'none' if callable(self.logger.experiment.name) else self.logger.experiment.name
            path = os.path.join(get_model_dir(), experiment_name)
            print("\nSaving predictions to ", path)
            os.makedirs(path, exist_ok=True)
            torch.save(C.cpu(), os.path.join(path, f"predictions_{mode}_{self.current_epoch}.pt"))
            torch.save(sentence_features.cpu(), os.path.join(path, f"sentence_embeddings_{mode}.pt"))
            torch.save(audio_features.cpu(), os.path.join(path, f"audio_embeddings_{mode}.pt"))
            np.save(os.path.join(path, f"paths_{mode}"), paths)
            np.save(os.path.join(path, f"captions_{mode}"), captions)
            print("\nSaving done!\n to ", path)


        top_one = C.topk(1, dim=1)[1]
        top_five = C.topk(5, dim=1)[1]
        top_ten = C.topk(10, dim=1)[1]

        target = torch.arange(len(audio_features)//n_captions)
        target = torch.repeat_interleave(target, n_captions)

        r_1 = (top_one == target[:, None]).float().sum(axis=1).mean().item()
        r_5 = (top_five == target[:, None]).float().sum(axis=1).mean().item()
        r_10 = (top_ten == target[:, None]).float().sum(axis=1).mean().item()

        AP = 1 / ((top_ten == target[:, None]).float().argmax(dim=1) + 1)
        AP[~(top_ten == target[:, None]).any(dim=1)] = 0
        mAP = AP.mean().item()

        self.log(f'{mode}/R@1', r_1, add_dataloader_idx=False, sync_dist=True)
        self.log(f'{mode}/R@5', r_5, add_dataloader_idx=False, sync_dist=True)
        self.log(f'{mode}/R@10', r_10, add_dataloader_idx=False, sync_dist=True)
        self.log(f'{mode}/mAP@10', mAP, add_dataloader_idx=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, mode='test')

    def test_step_end(self, *args, **kwargs):
        return self.validation_step_end(*args, **kwargs, mode='test')

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, mode='test')

    def predict_step(self, batch, batch_idx, **kwargs):
        return self.validation_step(batch, batch_idx, mode='predict')

    def on_predict_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, mode='predict')

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters())
        return {
            "optimizer": optimizer
        }

    @audio_retrieval.capture
    def update_scalars(self, batch_idx, hard_steps=True):
        epoch = self.current_epoch + (batch_idx / self.trainer.num_training_batches)
        if hard_steps:
            epoch = epoch // 1

        # weight decay - keep constant
        self.log('trainer/weight_decay', self.trainer.optimizers[0].param_groups[0]['weight_decay'])

        # learning rate
        update_lr(self.optimizers(use_pl_optimizer=False), epoch)
        self.log('trainer/lr', self.trainer.optimizers[0].param_groups[0]['lr'])


@audio_retrieval.capture
def update_lr(optimizer, epoch, lr, min_lr, warmup_length, rampdown_start, rampdown_stop, max_epochs, warmup_type='linear', rampdown_type='cosine'):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if rampdown_stop <= 0:
        rampdown_stop = max_epochs

    if epoch < warmup_length:
        if warmup_type == 'linear':
            lr = lr * epoch / warmup_length
        elif warmup_type == 'exp':
            epoch = np.clip(epoch, 0.5, warmup_length)
            phase = 1.0 - epoch / warmup_length
            lr = lr * float(np.exp(-5.0 * phase * phase))
        else:
            raise NotImplementedError
    elif epoch < rampdown_start:
        lr = lr
    elif epoch < rampdown_stop:

        if rampdown_type == 'cosine':
            offset = rampdown_start
            lr = min_lr + (lr - min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (epoch - offset) / (rampdown_stop - offset)))
        elif rampdown_type.startswith('step'):
            distance, factor = rampdown_type.split('_')[1:]
            distance, factor = int(distance), float(factor)
            steps = epoch // distance
            lr = lr*(factor**steps)
            lr = max(lr, min_lr)
        elif rampdown_type == 'linear':
            e = epoch - rampdown_start
            m = rampdown_stop - rampdown_start
            lr -= (lr - min_lr) * (e / m)
            lr = max(lr, min_lr)
        else:
            raise NotImplementedError
    else:
        lr = min_lr

    for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr


@audio_retrieval.capture
def get_optimizer(parameters, lr, beta1, beta2, eps, weight_decay, amsgrad, adamw):
    if adamw:
        return torch.optim.AdamW(parameters, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    else:
        return torch.optim.Adam(parameters, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


@audio_retrieval.capture
def get_trainer(wandb_logger, max_epochs, max_samples_per_epoch, gpus, half_precision, accelerator, enable_checkpointing, fast_dev_run, accumulate_grad_batches,
                gradient_clip_val, strategy, num_nodes, _config):
    if max_samples_per_epoch is None:
        max_steps_per_epoch = 0
    else:
        max_steps_per_epoch = max_samples_per_epoch // _config['data_loader']['batch_size']
    kwargs = {}
    if fast_dev_run:
       if max_steps_per_epoch == 0:
           print('Using fast_dev_run with max_epochs=5')
           max_steps_per_epoch = 5
       else:
           raise ValueError('Cannot use fast_dev_run with max_samples_per_epoch')
    return pl.Trainer(
        devices=gpus,
        num_nodes=num_nodes,
        accelerator=accelerator,
        val_check_interval=1.0,
        enable_checkpointing=enable_checkpointing > 0,
        logger=wandb_logger,
        max_epochs=max_epochs,
        callbacks=get_callbacks(wandb_logger),
        auto_select_gpus=True,
        precision=16 if half_precision else 32,
        limit_train_batches=1.0 if max_steps_per_epoch <= 0 else max_steps_per_epoch // gpus,
        limit_val_batches=1.0 ,
        reload_dataloaders_every_n_epochs=1 if max_steps_per_epoch > 0 else 0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        fast_dev_run=False,
        strategy=strategy,
        **kwargs
    )

@audio_retrieval.capture
def get_callbacks(wandb_logger, monitor, enable_checkpointing):
    callbacks = []

    if wandb_logger == None: # or callable(wandb_logger.experiment.name):
         print('No logger; skipping checkpoints')
    #     return []
    else:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    if enable_checkpointing:
        monitor = f'val/{monitor}'
        experiment_name = 'none' if wandb_logger is None or callable(wandb_logger.experiment.name) else wandb_logger.experiment.name
        save_dir = os.path.join(get_model_dir(), experiment_name)
        os.makedirs(save_dir, exist_ok=True)

        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir,
                monitor=monitor,
                mode='max',
                save_top_k=1,
                every_n_epochs=enable_checkpointing,
                save_last=True,
                auto_insert_metric_name=False,
                filename='epoch_{epoch}-{' + f'{monitor}' + '}'
            )
        )

    return callbacks


@audio_retrieval.command
def print_lr(max_epochs):
    import matplotlib.pyplot as plt
    import numpy as np

    lrs = [update_lr(torch.optim.Adam(torch.nn.Linear(1,1).parameters()), x) for x in np.linspace(0, max_epochs, 10000)]

    plt.plot(np.linspace(0, max_epochs, 10000), lrs)
    plt.show()


@audio_retrieval.command
def test_dataset(log_db_name, train_on, resume_training, eval_on_test, monitor, translate_augment_p, translate_augment_languages, eda_p, eda_p_swap, eda_p_ins, eda_p_del, eda_p_syn, _config=None):

    # create the data module
    print(f'Loading {train_on} data sets...')
    train = get_data_set(train_on.split('_')[0], 'train')

    # train = torch.utils.data.Subset(train, [2356*64 + i for i in range(1024)])
    dl = get_train_data_loader(train, collate_fun=collate_fun, shuffle=False)

    from tqdm import tqdm
    i = 0
    try:
        for i, d in enumerate(tqdm(dl)):
            pass
    except Exception as e:
        print(e)
        print(i)

    print('Done!')


def multiprocessing_run(rank, word_size, pernode=None):
    import socket
    print("rank ", rank, os.getpid(), "hash=", hash("kk test"), " on node ", socket.gethostname())
    print("word_size ", word_size)
    if pernode is None:
        pernode = word_size
    print("Tasks per node = ", pernode)
 
    os.environ['NODE_RANK'] = str(rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(",")[
        rank%pernode]
    print("Sat os.environ['CUDA_VISIBLE_DEVICES']=", os.environ['CUDA_VISIBLE_DEVICES'])
    # torch.cuda.set_device(int(os.environ['CUDA_VISIBLE_DEVICES'].split(",")[
    #     rank]))
    argv = sys.argv
    if rank != 0:
        print(f"Unobserved {os.getpid()} with rank {rank}")
        argv = argv + ["-u"]  # only rank 0 is observed
    if "with" not in argv:
        argv = argv + ["with"]

    argv = argv + \
        [f"num_nodes={word_size}", f"strategy=ddp"]
    print(argv)

    @audio_retrieval.main
    def main():
        return run_one()

    audio_retrieval.run_commandline(argv)




if __name__ == '__main__':
    # set DDP=2 forks two processes to run on two GPUs
    # the environment variable "DDP" define the number of processes to fork
    # With two 2x 2080ti you can train the full model to .47 in around 24 hours
    # you may need to set NCCL_P2P_DISABLE=1
    global word_size
    word_size = os.environ.get("DDP", None)
    DDP_SLURM = os.environ.get("DDP_SLURM", None)
    if DDP_SLURM:
        print("\n***SLLURM DDP MODE***\n\n")
        if "SLURM_NTASKS" in os.environ:
            del os.environ["SLURM_NTASKS"]
        if "SLURM_JOB_NAME" in os.environ:
            del os.environ["SLURM_JOB_NAME"]
        word_size = int(os.environ.get("WORLD_SIZE", None))
        print("word_size = ", word_size)
        pernode = int(os.environ.get("SLURM_NTASKS_PER_NODE", None))
        print("pernode = ", pernode)
        rank = int(os.environ.get("SLURM_PROCID", None))
        print("rank = ", rank)
        os.environ['PL_IN_DDP_SUBPROCESS'] = '1'
        print("I'm runing  with, pid=", os.getpid())
        multiprocessing_run(rank, word_size, pernode)
        exit(0)
        
    if word_size:
        import random
        if "SLURM_NTASKS" in os.environ:
            del os.environ["SLURM_NTASKS"]
        if "SLURM_JOB_NAME" in os.environ:
            del os.environ["SLURM_JOB_NAME"]
        word_size = int(word_size)
        print(f"\n\nDDP TRAINING WITH WORD_SIZE={word_size}\n\n")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        # plz no collisions
        os.environ['MASTER_PORT'] = f"{9999 + random.randint(0, 9999)}"
        os.environ['PL_IN_DDP_SUBPROCESS'] = '1'
        os.environ['WORLD_SIZE'] = str(word_size)
        for rank in range(word_size):
            pid = os.fork()
            if pid == 0:
                print("Child Forked, pid=", os.getpid())
                multiprocessing_run(rank, word_size)
                exit(0)

        pid, exit_code = os.wait()
        print(pid, exit_code)
        exit(0)

print("__main__ is running pid", os.getpid(), "in module main: ", __name__)




@audio_retrieval.automain
def main():
    return run_one()


