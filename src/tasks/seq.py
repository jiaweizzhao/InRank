from typing import Any, List

import torch
import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection

from einops import rearrange

from omegaconf import OmegaConf

from src.utils.utils import get_logger
from src.optim.param_grouping import group_parameters_for_optimizer
from src.utils.checkpoint import load_checkpoint

from src.models.layers.in_rank_efficient import InRankEfficient

logger = get_logger(__name__)


class SequenceModel(LightningModule):

    def __init__(self, cfg, model_cfg=None):
        """If model_cfg is passed, it will take precedence over cfg.model
        """
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model_cfg = model_cfg or self.cfg.model

        self.instantiate_datamodule()
        self.instantiate_model()
        self.warmstart()
        self.instantiate_loss()
        self.instantiate_metrics()

    def instantiate_datamodule(self):
        logger.info(f"Instantiating datamodule <{self.cfg.datamodule._target_}>")
        # Calling this self.datamodule will mess with PL since it also assigns self.datamodule
        self._datamodule: LightningDataModule = hydra.utils.instantiate(self.cfg.datamodule)
        self._datamodule.prepare_data()
        self._datamodule.setup()

    def instantiate_model(self):
        if hasattr(self._datamodule, 'num_classes'):
            self.model_cfg.num_classes = self._datamodule.num_classes
        if (hasattr(self._datamodule, 'vocab_size')
            and self.model_cfg.get('embedding_cfg', None) is not None):
            self.model_cfg.embedding_cfg.num_embeddings = self._datamodule.vocab_size
        logger.info(f"Instantiating model <{self.model_cfg._target_}>")
        recursive = getattr(self.model_cfg, '_recursive_', False)
        self.model = hydra.utils.instantiate(self.model_cfg, _recursive_=recursive)

    def instantiate_loss(self):
        loss_fn_cfg = self.cfg.train.get('loss_fn', {'_target_': 'torch.nn.CrossEntropyLoss'})
        self.loss_fn = hydra.utils.instantiate(loss_fn_cfg)
        loss_fn_val_cfg = self.cfg.train.get('loss_fn_val', loss_fn_cfg)
        self.loss_fn_val = hydra.utils.instantiate(loss_fn_val_cfg)

    def instantiate_metrics(self):
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        if 'eval' in self.cfg and 'metrics' in self.cfg.eval:
            metrics_cfg = self.cfg.eval.metrics
        else:
            metrics_cfg = {'acc': {'_target_': 'torchmetrics.Accuracy'}}
        metrics = MetricCollection({name: hydra.utils.instantiate(cfg)
                                    for name, cfg in metrics_cfg.items()})
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def warmstart(self):
        if self.cfg.train.get('warmstart', None) is not None:
            logger.info(f"Warm-starting with weights from {self.cfg.train.warmstart.path}")
            strict = self.cfg.train.warmstart.get('strict', True)
            state_dict = load_checkpoint(self.cfg.train.warmstart.path)
            if self.cfg.train.warmstart.get('post_process', None) is not None:
                state_dict = hydra.utils.instantiate(self.cfg.train.warmstart.post_process,
                                                     state_dict)
            load_return = self.model.load_state_dict(state_dict, strict=False)
            logger.info(load_return)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self, batch: Any, is_train=True):
        try:
            x, y, lengths = batch
        except ValueError:
            x, y = batch
            lengths = None
        output = self.forward(x) if lengths is None else self.forward(x, lengths=lengths)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets = self.step(batch, is_train=(phase == 'train'))
        with torch.no_grad():
            metrics = getattr(self, f'{phase}_metrics')(output, targets)
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "output": output, "targets": targets}

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='train')

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='val')

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        if 'optimizer_param_grouping' in self.cfg.train:  # Set zero weight decay for some params
            parameters = group_parameters_for_optimizer(self.model, self.cfg.train.optimizer,
                                                        **self.cfg.train.optimizer_param_grouping)
        else:
            # parameters = self.model.parameters()
            parameters = self.parameters() # [21-09-08] AG: this will train task specific parameters such as Retrieval head for AAN
        optimizer = hydra.utils.instantiate(self.cfg.train.optimizer, parameters)

        # Log optimizer info
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            logger.info(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')

        if 'scheduler' not in self.cfg.train:
            return optimizer
        else:
            # lr_scheduler should be called either every step (default) or every epoch
            lr_scheduler = hydra.utils.instantiate(self.cfg.train.scheduler, optimizer)
            return [optimizer], {'scheduler': lr_scheduler,
                                 'interval': self.cfg.train.get('scheduler_interval', 'step'),
                                 'monitor': self.cfg.train.get('scheduler_monitor', 'val/loss')}


class SequenceDualModel(SequenceModel):

    def step(self, batch: Any, is_train=True):
        x1, x2, y, lengths1, lengths2 = batch
        output = self.forward(x1, x2, lengths1=lengths1, lengths2=lengths2)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        output = torch.argmax(output, dim=1)
        return loss, output, y


class SequenceLMModel(SequenceModel):

    def instantiate_model(self):
        self.stage = self.model_cfg.get('stage', None)
        if self.stage != None:
            del self.model_cfg['stage']

        if (hasattr(self._datamodule, 'vocab_size')
            and self.model_cfg.get('embedding_cfg', None) is not None):
            self.model_cfg.embedding_cfg.num_embeddings = self._datamodule.vocab_size
        logger.info(f"Instantiating model <{self.model_cfg._target_}>")
        # Huggingface models need the config object to be instantiated first
        config = hydra.utils.instantiate(self.model_cfg.pop('config'), _recursive_=False)
        self.model = hydra.utils.instantiate(self.model_cfg, config, _recursive_=False)

        # baseline and stage_1 not execute the following code block
        if self.stage == "stage_2":
            if self.cfg.trainer.get('resume_from_checkpoint', None) is None:
                raise RuntimeError("resume_from_checkpoint should not be null when executing on stage_2")
            ckp = torch.load(self.cfg.trainer.get('resume_from_checkpoint'))
            self.group_current_modes = ckp.get('group_current_modes', None)

            if self.group_current_modes is not None:
                iterator_group_current_modes = iter(self.group_current_modes)
                for module in self.model.modules():
                    if isinstance(module, InRankEfficient):
                        module.resize(False, next(iterator_group_current_modes))
            else:
                raise RuntimeError("No group_current_modes when executing on stage_2 and the model should be pruned in stage_1")

    def step(self, batch: Any, is_train=True):
        x, y = batch
        output = self.forward(x).logits
        output = rearrange(output, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y
    
    def model_optimizer_pruning(self):
        # model pruning
        self.group_current_modes = []
        for module in self.model.modules():
            if isinstance(module, InRankEfficient):
                self.group_current_modes.append(module.resize(True))        
        
        # optimizer pruning    
        if 'optimizer_param_grouping' in self.cfg.train:
            parameters = group_parameters_for_optimizer(self.model, self.cfg.train.optimizer,
                                                        **self.cfg.train.optimizer_param_grouping)
            param_ls = []
            for param in parameters:
                param_ls.extend(param["params"])
            parameters = param_ls
        else:
            parameters = self.parameters()
            
        optimizer = self.optimizers().optimizer
        for i in range(len(optimizer.state_dict()['state'])):
            if optimizer.state_dict()['state'][i]['exp_avg'].size() != parameters[i].size():
                a, b = parameters[i].size()
                optimizer.state_dict()['state'][i]['exp_avg'] = optimizer.state_dict()['state'][i]['exp_avg'][:a, :b]
                optimizer.state_dict()['state'][i]['exp_avg_sq'] = optimizer.state_dict()['state'][i]['exp_avg_sq'][:a, :b]
#            print(optimizer.state_dict()['state'][5]['exp_avg'].size())
#            print(optimizer.state_dict()['state'][6]['exp_avg'].size())
#            print(optimizer.state_dict()['state'][7]['exp_avg'].size())
#            print(optimizer.state_dict()['state'][8]['exp_avg'].size())
#            print(optimizer.state_dict()['state'][9]['exp_avg'].size())

class ModelwNormalizer(SequenceModel):

    def instantiate_datamodule(self):
        super().instantiate_datamodule()
        # We need to register the datamodule's y_normalizer as sub-module
        # so that it gets moved to the current device.
        self.y_normalizer = self._datamodule.y_normalizer

    def step(self, batch: Any, is_train=True):
        x, y = batch
        output = self.forward(x)
        output = self.y_normalizer.decode(output)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y