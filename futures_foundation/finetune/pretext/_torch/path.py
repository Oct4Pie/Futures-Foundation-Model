"""Stage-3 fixed-wall-clock future-path supervision for transferable futures representations."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import load_mantis
from .common import (_apply_control, _gather_batch, BaseTrainer, encode_independent,
                     load_encoder_checkpoint, preprocess_windows, resolve_preprocessing)


PATH_QUANTILES = (0.25, 0.5, 0.75)
PATH_CLASSES = ('termination', 'continuation', 'reversal')


def wall_clock_steps(bar_ns, horizons_minutes):
    bar_ns = torch.as_tensor(bar_ns, dtype=torch.long)
    horizon_ns = torch.as_tensor(horizons_minutes, dtype=torch.long) * 60_000_000_000
    remainder = horizon_ns[None, :] % bar_ns[:, None]
    if torch.any(remainder):
        raise ValueError('every path horizon must be divisible by its stream bar duration')
    return horizon_ns[None, :] // bar_ns[:, None]


def path_targets(raw_context, raw_future, steps, *, context_minutes=60,
                 bar_ns=None, deadband_r=0.25):
    """Build raw-price future targets. Future rows are labels only and never enter the encoder."""
    if raw_context.ndim != 3 or raw_context.shape[1] < 5 or raw_future.ndim != 3:
        raise ValueError('path targets require aligned [B,C>=5,T] context/future tensors')
    batch, _, future_length = raw_future.shape
    steps = torch.as_tensor(steps, dtype=torch.long, device=raw_context.device)
    if steps.ndim != 2 or steps.shape[0] != batch or torch.any(steps < 1):
        raise ValueError('steps must be positive [B,horizons]')
    if int(steps.max()) > future_length:
        raise ValueError('path target steps exceed the reserved future')
    o, h, l, c = (raw_context[:, index, :] for index in range(4))
    prev = torch.cat((c[:, :1], c[:, :-1]), 1)
    tr = torch.maximum(h - l, torch.maximum((h - prev).abs(), (l - prev).abs()))
    scale = tr[:, -20:].mean(1).clamp_min(1e-6)
    base = c[:, -1]
    if bar_ns is None:
        context_steps = torch.full((batch,), min(raw_context.shape[-1] - 1, 60),
                                   dtype=torch.long, device=raw_context.device)
    else:
        context_ns = int(context_minutes) * 60_000_000_000
        bars = torch.as_tensor(bar_ns, dtype=torch.long, device=raw_context.device)
        if torch.any(context_ns % bars):
            raise ValueError('context direction duration must divide every stream bar duration')
        context_steps = (context_ns // bars).clamp(1, raw_context.shape[-1] - 1)
    old_index = raw_context.shape[-1] - 1 - context_steps
    old_close = c.gather(1, old_index[:, None]).squeeze(1)
    context_move_r = (base - old_close) / scale
    direction = torch.where(context_move_r >= 0, 1.0, -1.0)

    future_close = raw_future[:, 3, :]
    future_high, future_low = raw_future[:, 1, :], raw_future[:, 2, :]
    prev_future_close = torch.cat((base[:, None], future_close[:, :-1]), 1)
    future_returns = torch.log(future_close.clamp_min(1e-9) /
                               prev_future_close.clamp_min(1e-9))
    positions = torch.arange(future_length, device=raw_context.device)[None, :]
    vols, favorable, adverse, classes = [], [], [], []
    for horizon in range(steps.shape[1]):
        count = steps[:, horizon]
        valid = positions < count[:, None]
        realized_vol = torch.sqrt((future_returns.square() * valid).sum(1)).mul(100.0)
        if direction.ndim != 1:
            raise AssertionError('invalid direction shape')
        favorable_path = torch.where(direction[:, None] > 0,
                                     future_high - base[:, None], base[:, None] - future_low)
        adverse_path = torch.where(direction[:, None] > 0,
                                   base[:, None] - future_low, future_high - base[:, None])
        neg_inf = torch.full_like(favorable_path, -torch.inf)
        fav = torch.where(valid, favorable_path, neg_inf).max(1).values.clamp_min(0) / scale
        adv = torch.where(valid, adverse_path, neg_inf).max(1).values.clamp_min(0) / scale
        terminal_index = count - 1
        terminal = future_close.gather(1, terminal_index[:, None]).squeeze(1)
        terminal_r = direction * (terminal - base) / scale
        cls = torch.zeros(batch, dtype=torch.long, device=raw_context.device)
        cls[terminal_r > float(deadband_r)] = 1
        cls[terminal_r < -float(deadband_r)] = 2
        # An undefined/flat context is termination, not an arbitrarily assigned direction.
        cls[context_move_r.abs() < float(deadband_r)] = 0
        vols.append(torch.log1p(realized_vol))
        favorable.append(fav.clamp_max(20.0))
        adverse.append(adv.clamp_max(20.0))
        classes.append(cls)
    return {
        'log_vol': torch.stack(vols, 1),
        'favorable_r': torch.stack(favorable, 1),
        'adverse_r': torch.stack(adverse, 1),
        'path_class': torch.stack(classes, 1),
    }


class PathNetwork(nn.Module):
    outputs_per_horizon = 10  # log-vol + 3 favorable q + 3 adverse q + 3 class logits

    def __init__(self, C=5, horizons=3, model_id='paris-noah/Mantis-8M', model_version=None):
        super().__init__()
        self.encoder = load_mantis(model_id, model_version=model_version, device='cpu')
        hidden = getattr(self.encoder, 'hidden_dim', 256)
        embedding = hidden * C
        self.head = nn.Sequential(nn.Linear(embedding, embedding), nn.GELU(),
                                  nn.Linear(embedding, int(horizons) * self.outputs_per_horizon))
        self.C, self.horizons = C, int(horizons)

    def embed(self, x):
        return encode_independent(self.encoder, x)

    def forward(self, x):
        return self.head(self.embed(x)).view(-1, self.horizons, self.outputs_per_horizon)


def _monotone_quantiles(raw):
    return torch.cumsum(F.softplus(raw), dim=-1)


def _pinball(prediction, target, quantiles=PATH_QUANTILES):
    q = prediction.new_tensor(quantiles)
    error = target[..., None] - prediction
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def path_loss(output, target, *, vol_weight=1.0, excursion_weight=1.0,
              class_weight=1.0):
    log_vol = output[..., 0]
    favorable = _monotone_quantiles(output[..., 1:4])
    adverse = _monotone_quantiles(output[..., 4:7])
    logits = output[..., 7:10]
    vol = F.smooth_l1_loss(log_vol.float(), target['log_vol'].float())
    excursions = (_pinball(favorable.float(), target['favorable_r'].float()) +
                  _pinball(adverse.float(), target['adverse_r'].float()))
    classification = F.cross_entropy(logits.reshape(-1, 3).float(),
                                     target['path_class'].reshape(-1))
    total = float(vol_weight) * vol + float(excursion_weight) * excursions
    total = total + float(class_weight) * classification
    return total, {'vol_loss': vol, 'excursion_loss': excursions, 'class_loss': classification}


class _PathTrainer(BaseTrainer):
    def __init__(self, big, tr, va, *, seq=256, path_horizons_minutes=(60, 180, 360),
                 path_context_minutes=60, path_max_future_bars=360,
                 path_vol_weight=1.0, path_excursion_weight=1.0, path_class_weight=1.0,
                 feature_anchor_weight=0.1, model_id='paris-noah/Mantis-8M', model_version=None,
                 backbone_ckpt=None, preprocessing=None, compile_model=False,
                 stream_bar_ns=None, objective_row_bounds=None, **base):
        super().__init__(big, tr, va, **base)
        self.seq = int(seq)
        self.horizons_minutes = tuple(int(x) for x in path_horizons_minutes)
        if not self.horizons_minutes or tuple(sorted(set(self.horizons_minutes))) != self.horizons_minutes:
            raise ValueError('path horizons must be unique and increasing')
        self.context_minutes, self.max_future = int(path_context_minutes), int(path_max_future_bars)
        self.vol_weight = float(path_vol_weight)
        self.excursion_weight = float(path_excursion_weight)
        self.class_weight = float(path_class_weight)
        if min(self.vol_weight, self.excursion_weight, self.class_weight) < 0:
            raise ValueError('path loss weights must be nonnegative')
        self.feature_anchor_weight = float(feature_anchor_weight)
        self.model_id, self.model_version = model_id, model_version
        self.backbone_ckpt, self.compile_model = backbone_ckpt, compile_model
        self.preprocessing = resolve_preprocessing(preprocessing, backbone_ckpt)
        self.C = int(self.big_t.shape[1])
        bars = torch.as_tensor(stream_bar_ns, dtype=torch.long, device=self.dev)
        if self.tr_groups is None or self.va_groups is None or len(bars) != len(self.tr_groups):
            raise ValueError('path training requires bar duration for every balanced stream group')
        self.stream_bar_ns = bars
        all_steps = wall_clock_steps(bars.cpu(), self.horizons_minutes).to(self.dev)
        if int(all_steps.max()) > self.max_future:
            raise ValueError('path_max_future_bars does not reserve the declared horizons')
        self.steps_by_group = all_steps
        segments = torch.as_tensor(objective_row_bounds, dtype=torch.long, device=self.dev)
        if segments.ndim != 2 or segments.shape[1] != 2:
            raise ValueError('path training requires stream/contract objective boundaries')
        self.objective_row_bounds = segments
        self.tr, self.tr_groups = self._filter_future_safe(self.tr, self.tr_groups, 'train')
        self.va, self.va_groups = self._filter_future_safe(self.va, self.va_groups, 'validation')

    def _filter_future_safe(self, starts, bounds, name):
        """Keep contexts whose complete wall-clock label stays inside split and contract."""
        pieces, new_bounds, cursor = [], [], 0
        segment_ends = self.objective_row_bounds[:, 1].contiguous()
        for group, (lo, hi) in enumerate(bounds.tolist()):
            values = starts[lo:hi]
            max_steps = int(self.steps_by_group[group, -1])
            # The latest seq-valid start identifies this split's exclusive row boundary. This is
            # deliberately conservative when the final eligible context precedes a market gap.
            split_end = int(values[-1]) + self.seq
            segment_index = torch.searchsorted(segment_ends, values, right=True)
            if torch.any(segment_index >= len(self.objective_row_bounds)):
                raise ValueError(f'{name} path start is outside objective segments')
            segment_end = self.objective_row_bounds[segment_index, 1]
            keep = values + self.seq + max_steps <= torch.minimum(
                segment_end, torch.full_like(segment_end, split_end))
            selected = values[keep]
            if len(selected) == 0:
                raise ValueError(f'{name} stream group {group} has no future-safe path windows')
            pieces.append(selected)
            new_bounds.append((cursor, cursor + len(selected)))
            cursor += len(selected)
        return torch.cat(pieces), torch.as_tensor(new_bounds, dtype=torch.long, device=self.dev)

    def build_net(self):
        net = PathNetwork(self.C, len(self.horizons_minutes), self.model_id,
                          self.model_version).to(self.dev)
        if self.backbone_ckpt:
            load_encoder_checkpoint(net.encoder, self.backbone_ckpt, model_id=self.model_id,
                                    model_version=self.model_version, expected_channels=self.C)
        self.teacher = None
        if self.feature_anchor_weight > 0:
            self.teacher = load_mantis(self.model_id, model_version=self.model_version, device='cpu')
            self.teacher.load_state_dict(net.encoder.state_dict())
            self.teacher = self.teacher.to(self.dev).eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad = False
        if self.compile_model and hasattr(torch, 'compile'):
            net = torch.compile(net)
        self.net = net

    def make_batch(self, starts):
        indices, groups = self.sample_start_indices(starts, return_groups=True)
        context = _gather_batch(self.big_t, starts, indices, self.seq)
        selected_starts = starts[indices]
        max_steps = self.steps_by_group[groups, -1]
        offsets = torch.arange(self.max_future, device=self.dev)[None, :]
        # Duplicate only unused tail positions so a mixed-timeframe batch stays rectangular;
        # path_targets masks every position at/after the sample's exact horizon.
        safe_offsets = torch.minimum(offsets, max_steps[:, None] - 1)
        rows = selected_starts[:, None] + self.seq + safe_offsets
        future = self.big_t[rows].permute(0, 2, 1).contiguous()
        bar_ns = self.stream_bar_ns[groups]
        target = path_targets(context, future, self.steps_by_group[groups],
                              context_minutes=self.context_minutes, bar_ns=bar_ns)
        clean = preprocess_windows(context, self.preprocessing)
        return _apply_control(clean, self.control), clean, target

    def compute_loss(self, batch):
        model_context, clean_context, target = batch
        loss, _ = path_loss(self.net(model_context), target, vol_weight=self.vol_weight,
                            excursion_weight=self.excursion_weight,
                            class_weight=self.class_weight)
        if self.teacher is not None:
            with torch.no_grad():
                teacher = encode_independent(self.teacher, clean_context)
            loss = loss + self.feature_anchor_weight * F.mse_loss(
                self.net.embed(clean_context).float(), teacher.float())
        return loss

    @torch.no_grad()
    def val_eval(self):
        self.net.eval(); totals = {'loss': 0.0, 'vol_loss': 0.0, 'excursion_loss': 0.0,
                                   'class_loss': 0.0, 'class_acc': 0.0}
        batches = min(self.val_batches or 20, max(1, len(self.va) // self.batch))
        with self.fixed_validation_rng():
            for _ in range(batches):
                model_context, clean_context, target = self.make_batch(self.va)
                output = self.net(model_context)
                loss, parts = path_loss(output, target, vol_weight=self.vol_weight,
                                        excursion_weight=self.excursion_weight,
                                        class_weight=self.class_weight)
                totals['loss'] += float(loss)
                for name in ('vol_loss', 'excursion_loss', 'class_loss'):
                    totals[name] += float(parts[name])
                totals['class_acc'] += float(
                    (output[..., 7:10].argmax(-1) == target['path_class']).float().mean())
            std = float(self.net.embed(clean_context).std(0).mean())
        self.net.train()
        return totals.pop('loss') / batches, {name: value / batches for name, value in totals.items()} | {'std': std}

    def _resume_signature(self):
        return {**super()._resume_signature(), 'objective': 'path_core_v1', 'seq': self.seq,
                'path_horizons_minutes': self.horizons_minutes,
                'path_context_minutes': self.context_minutes,
                'path_max_future_bars': self.max_future,
                'path_vol_weight': self.vol_weight,
                'path_excursion_weight': self.excursion_weight,
                'path_class_weight': self.class_weight,
                'feature_anchor_weight': self.feature_anchor_weight}


def train_ssl_path(big, train_starts, val_starts, *, seq=256,
                   path_horizons_minutes=(60, 180, 360), path_context_minutes=60,
                   path_max_future_bars=360, path_vol_weight=1.0,
                   path_excursion_weight=1.0, path_class_weight=1.0,
                   feature_anchor_weight=0.1, epochs=60, steps_per_epoch=200, batch=128,
                   lr=1e-4, weight_decay=0.05, patience=8, device=None,
                   model_id='paris-noah/Mantis-8M', model_version=None, backbone_ckpt=None,
                   compile_model=False, control='real', seed=0, amp_dtype='bf16', verbose=True,
                   ckpt_path=None, resume=False, freeze_encoder_layers=0, **extra):
    return _PathTrainer(
        big, train_starts, val_starts, seq=seq, path_horizons_minutes=path_horizons_minutes,
        path_context_minutes=path_context_minutes, path_max_future_bars=path_max_future_bars,
        path_vol_weight=path_vol_weight, path_excursion_weight=path_excursion_weight,
        path_class_weight=path_class_weight, feature_anchor_weight=feature_anchor_weight,
        model_id=model_id, model_version=model_version, backbone_ckpt=backbone_ckpt,
        preprocessing=extra.get('preprocessing'), compile_model=compile_model,
        stream_bar_ns=extra.get('stream_bar_ns'), epochs=epochs,
        objective_row_bounds=extra.get('objective_row_bounds'),
        steps_per_epoch=steps_per_epoch, batch=batch, lr=lr, weight_decay=weight_decay,
        patience=patience, device=device, seed=seed, grad_clip=1.0, amp_dtype=amp_dtype,
        verbose=verbose, control=control, ckpt_path=ckpt_path, resume=resume,
        freeze_encoder_layers=freeze_encoder_layers,
        train_group_bounds=extra.get('train_group_bounds'),
        val_group_bounds=extra.get('val_group_bounds'), val_batches=extra.get('val_batches'),
    ).fit()
