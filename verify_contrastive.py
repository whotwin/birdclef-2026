import sys, yaml, os, torch
sys.path.insert(0, 'src_contrastive_learn')
from dataset import ContrastiveAudioDataset
from augmentation import get_contrastive_augmentation
from train import contrastive_collate_fn
from encoder import SimCLREncoder
from loss import NTXentLoss

cfg = yaml.safe_load(open('src_contrastive_learn/config.yaml', encoding='utf-8'))
print('n_mels=%d, weight_decay=%s' % (cfg['data']['n_mels'], cfg['training']['weight_decay']))

aug = get_contrastive_augmentation(
    noise_std=cfg['augmentation']['noise_std'],
    volume_jitter=cfg['augmentation']['volume_jitter'],
)
ds = ContrastiveAudioDataset(
    csv_path=cfg['data']['csv_path'],
    audio_dir=cfg['data']['audio_dir'],
    sr=cfg['data']['sr'],
    duration=cfg['data']['duration'],
    n_mels=cfg['data']['n_mels'],
    fmin=cfg['data']['fmin'],
    fmax=cfg['data']['fmax'],
    augmentation=aug,
    soundscapes_dir=cfg['data'].get('soundscapes_dir'),
    n_samples_per_soundscape=cfg['data'].get('n_samples_per_soundscape', 2),
)
print('Dataset: total=%d, audio=%d, soundscape=%d' % (len(ds), ds._n_audio, ds._n_soundscape))

v1, v2 = ds[0]
print('train_audio view1 shape:', v1.shape)

if ds._n_soundscape > 0:
    v1s, v2s = ds[ds._n_audio]
    print('soundscape view1 shape:', v1s.shape)

batch = [ds[i] for i in range(4)]
v1b, v2b = contrastive_collate_fn(batch)
print('Batched view1 shape:', v1b.shape)
print('Batched view2 shape:', v2b.shape)

assert v1b.shape[1] == 1, 'channel dim should be 1, got %d' % v1b.shape[1]
assert v1b.shape[2] == 128, 'n_mels should be 128, got %d' % v1b.shape[2]
print('in_chans=1, n_mels=128 verified!')

model = SimCLREncoder(
    backbone_name=cfg['model']['backbone'],
    projection_dim=cfg['model']['projection_dim'],
    pretrained=False,
)
z1 = model(v1b)
z2 = model(v2b)
print('z1 shape:', z1.shape)
print('z2 shape:', z2.shape)
assert z1.shape == (4, cfg['model']['projection_dim'])
assert z2.shape == (4, cfg['model']['projection_dim'])

criterion = NTXentLoss(temperature=cfg['training']['temperature'])
loss = criterion(z1, z2)
print('NTXentLoss: %.4f' % loss.item())

print('ALL CHECKS PASSED!')
