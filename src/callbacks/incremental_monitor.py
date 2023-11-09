from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger

from src.models.layers.in_rank_efficient import InRankEfficient

def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )

class LogIncrementalMLP(Callback):
    
    def __init__(self):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        
        # params
        log_batch_gap = 2000
        num_visualized_modes = 20
        
        # report gap
        if batch_idx % log_batch_gap != 0:
            return
        
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
                       
        model = pl_module.model
        # list all modules
        for name, module in model.named_modules():
            # check if module is a linear layer
            if isinstance(module, InRankEfficient):
                current_modes = module.current_modes
                explained_ratio = module.current_explained_ratio
                current_s_vector = module.current_s_vector
                
                experiment.log({f"{name}/current_modes": current_modes})
               
                experiment.log({f"{name}/current_explained_ratio": explained_ratio})
               
                if current_s_vector is not None:
                   for index, current_s in enumerate(current_s_vector):
                       if index < num_visualized_modes:
                           experiment.log({f"{name}/singular_value_{index}": current_s})
                       else:
                           break
