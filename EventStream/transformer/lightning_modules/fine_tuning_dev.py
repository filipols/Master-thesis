import dataclasses
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import wandb
import lightning as L
import omegaconf
import torch
import torch.multiprocessing
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)
from transformers import get_polynomial_decay_schedule_with_warmup

from ...data.config import (
    PytorchDatasetConfig,
    SeqPaddingSide,
    SubsequenceSamplingStrategy,
)
from ...data.pytorch_dataset import PytorchDataset
from ...utils import hydra_dataclass, task_wrapper
from ..config import OptimizationConfig, StructuredTransformerConfig, MetricsConfig
from ..fine_tuning_model import ESTForStreamClassification
from ..model_output import StreamClassificationModelOutput
from ..utils import str_summary
from .generative_modeling import ESTForGenerativeSequenceModelingLM





@hydra_dataclass
class FinetuneConfig:
    
    load_from_model_dir: str | Path | None = omegaconf.MISSING
    task_df_name: str | None = omegaconf.MISSING
    strategy: bool = False
    pretrained_weights_fp: Path | str | None = "${load_from_model_dir}/pretrained_weights"


    experiment_dir: str | Path | None = "${load_from_model_dir}/finetuning"
    
    save_dir: str | None = (
        "${experiment_dir}/${task_df_name}"
    )
        


    wandb_logger_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "${task_df_name}_finetuning",
            "project": None,
            "team": None,
            "log_model": True,
            "do_log_graph": True,
        }
    )

    wandb_experiment_config_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "save_dir": "${save_dir}",
        }
    )

    do_overwrite: bool = False
    seed: int = 1

    # Config override parameters
    config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            **{k: None for k in StructuredTransformerConfig().to_dict().keys()},
            "task_specific_params": {
                "pooling_method": "last",
                "num_samples": None,
            },
        }
    )
    optimization_config: OptimizationConfig = dataclasses.field(default_factory=lambda: OptimizationConfig())
    data_config: dict[str, Any] | None = dataclasses.field(
        default_factory=lambda: {
            **{k: None for k in PytorchDatasetConfig().to_dict().keys()},
            "subsequence_sampling_strategy": SubsequenceSamplingStrategy.TO_END,
            "seq_padding_side": SeqPaddingSide.RIGHT,
            "task_df_name": "${task_df_name}",
            "train_subset_size": "FULL",
            "train_subset_seed": 1,
            "save_dir": "test",
        }
    )

    trainer_config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": "auto",
            "detect_anomaly": False,
            "default_root_dir": "${save_dir}/model_checkpoints",
            "log_every_n_steps": 10,
        }
    )

    metrics_config: MetricsConfig = dataclasses.field(default_factory=lambda: MetricsConfig())
    do_use_filesystem_sharing: bool = True

    def __post_init__(self):

        if self.strategy:
            self.experiment_dir = Path(self.load_from_model_dir)
        else:
            self.experiment_dir = Path(self.load_from_model_dir) / "finetuning"

        self.save_dir = Path(self.experiment_dir) / self.task_df_name
        
        match self.save_dir:
            case str():
                self.save_dir = Path(self.save_dir)
            case Path():
                pass
            case _:
                raise TypeError(
                    f"`save_dir` must be a str or path! Got {type(self.save_dir)}({self.save_dir})"
                )

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        elif not self.save_dir.is_dir():
            raise FileExistsError(f"{self.save_dir} is not a directory!")

        if self.load_from_model_dir in (omegaconf.MISSING, None, "skip"):
            self.config = StructuredTransformerConfig(
                **{k: v for k, v in self.config.items() if v is not None}
            )
            self.data_config = PytorchDatasetConfig(**self.data_config)
            return

        match self.pretrained_weights_fp:
            case "skip" | None | Path():
                pass
            case str():
                self.pretrained_weights_fp = Path(self.pretrained_weights_fp)
            case _:
                raise TypeError(
                    "`pretrained_weights_fp` must be a str or path! Got "
                    f"{type(self.pretrained_weights_fp)}({self.pretrained_weights_fp})"
                )

        match self.load_from_model_dir:
            case str():
                self.load_from_model_dir = Path(self.load_from_model_dir)
            case Path():
                pass
            case _:
                raise TypeError(
                    "`load_from_model_dir` must be a str or path! Got "
                    f"{type(self.load_from_model_dir)}({self.load_from_model_dir})"
                )

        # convert data_config.save_dir to Path
        match self.data_config["save_dir"]:
            case str():
                self.data_config["save_dir"] = Path(self.data_config["save_dir"])
            case Path():
                pass
            case _:
                raise TypeError(
                    "`data_config.save_dir` must be a str or path! Got "
                    f"{type(self.data_config.save_dir)}({self.data_config.save_dir})"
                )

        if (
            self.data_config.get("train_subset_size", "FULL") != "FULL"
            and self.data_config.get("train_subset_seed", None) is None
        ):
            self.data_config["train_subset_seed"] = int(random.randint(1, int(1e6)))
            print(
                f"WARNING: train_subset_size={self.data_config.train_subset_size} but "
                f"seed is unset. Setting to {self.data_config['train_subset_seed']}"
            )

        data_config_fp = self.load_from_model_dir / "data_config.json"
        print(f"Loading data_config from {data_config_fp}")
        reloaded_data_config = PytorchDatasetConfig.from_json_file(data_config_fp)
        reloaded_data_config.task_df_name = self.task_df_name

        print('reloaded_data_config', reloaded_data_config)

        for param, val in self.data_config.items():
            if val is None:
                continue
            if param == "task_df_name":
                if val != self.task_df_name:
                    print(
                        f"WARNING: task_df_name is set in data_config_overrides to {val}! "
                        f"Original is {self.task_df_name}. Ignoring data_config..."
                    )
                continue
            print(f"Overwriting {param} in data_config from {getattr(reloaded_data_config, param)} to {val}")
            setattr(reloaded_data_config, param, val)

        self.data_config = reloaded_data_config

        config_fp = self.load_from_model_dir / "config.json"
        print(f"Loading config from {config_fp}")
        reloaded_config = StructuredTransformerConfig.from_json_file(config_fp)

        for param, val in self.config.items():
            if val is None:
                continue
            print(f"Overwriting {param} in config from {getattr(reloaded_config, param)} to {val}")
            setattr(reloaded_config, param, val)

        self.config = reloaded_config

        reloaded_pretrain_config = OmegaConf.load(self.load_from_model_dir / "pretrain_config.yaml")
        if self.wandb_logger_kwargs.get("project", None) is None:
            print(f"Setting wandb project to {reloaded_pretrain_config.wandb_logger_kwargs.project}")
            self.wandb_logger_kwargs["project"] = reloaded_pretrain_config.wandb_logger_kwargs.project


@task_wrapper
def train(cfg: FinetuneConfig):
    """Runs the end to end training procedure for the fine-tuning model.

    Args:
        cfg: The fine-tuning configuration object specifying the cohort and task for which and model from
            which you wish to fine-tune.
    """

    L.seed_everything(cfg.seed)
    if cfg.do_use_filesystem_sharing:
        torch.multiprocessing.set_sharing_strategy("file_system")

    train_pyd = PytorchDataset(cfg.data_config, split="train")
    tuning_pyd = PytorchDataset(cfg.data_config, split="tuning")

    config = cfg.config
    data_config = cfg.data_config
    optimization_config = cfg.optimization_config

    config.set_to_dataset(train_pyd)
    optimization_config.set_to_dataset(train_pyd)


    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
        print("Saving config files...")
        config_fp = cfg.save_dir / "config.json"
        if config_fp.exists() and not cfg.do_overwrite:
            raise FileExistsError(f"{config_fp} already exists!")
        else:
            print(f"Writing to {config_fp}")
            config.to_json_file(config_fp)

        data_config.to_json_file(cfg.save_dir / "data_config.json", do_overwrite=cfg.do_overwrite)
        optimization_config.to_json_file(
            cfg.save_dir / "optimization_config.json", do_overwrite=cfg.do_overwrite
        )

    # Model
    model_params = dict(config=config, optimization_config=optimization_config,metrics_config=cfg.metrics_config)
    
    if cfg.pretrained_weights_fp is not None:
        model_params["pretrained_weights_fp"] = cfg.pretrained_weights_fp

    LM = ESTForGenerativeSequenceModelingLM(**model_params)


    # Setting up torch dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_pyd,
        batch_size=optimization_config.batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=train_pyd.collate,
        shuffle=True,
    )
    tuning_dataloader = torch.utils.data.DataLoader(
        tuning_pyd,
        batch_size=optimization_config.validation_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=tuning_pyd.collate,
        shuffle=False,
    )
    # Setting up model configurations
    # This will track the learning rate value as it updates through warmup and decay.
    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        filename="{epoch}-{val_loss:.2f}-best_model",
        monitor="tuning_loss",
        mode="min",
        save_top_k=3,
    )
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        checkpoint_callback,
    ]
    if optimization_config.patience is not None:
        callbacks.append(
            EarlyStopping(monitor="tuning_loss", mode="min", patience=optimization_config.patience)
        )

    checkpoints_dir = cfg.save_dir / "model_checkpoints"
    checkpoints_dir.mkdir(parents=False, exist_ok=True)

    trainer_kwargs = dict(
        **cfg.trainer_config,
        max_epochs=optimization_config.max_epochs,
        callbacks=callbacks,
        strategy="ddp_find_unused_parameters_true"
    )

    if cfg.wandb_logger_kwargs.get("name", None):
        if "do_log_graph" in cfg.wandb_logger_kwargs:
            do_log_graph = cfg.wandb_logger_kwargs.pop("do_log_graph")
        else:
            do_log_graph = False

        wandb_logger = WandbLogger(
            **{k: v for k, v in cfg.wandb_logger_kwargs.items() if v is not None},
            save_dir=cfg.save_dir,
        )
        
        if os.environ.get("LOCAL_RANK", "0") == "0":
            if do_log_graph:
                # Watching the model naturally tracks parameter values and gradients.
                wandb_logger.watch(LM, log="all", log_graph=True)

            if cfg.wandb_experiment_config_kwargs:
                wandb_logger.experiment.config.update(cfg.wandb_experiment_config_kwargs)

        trainer_kwargs["logger"] = wandb_logger

    if (optimization_config.gradient_accumulation is not None) and (
        optimization_config.gradient_accumulation > 1
    ):
        trainer_kwargs["accumulate_grad_batches"] = optimization_config.gradient_accumulation

    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader)

    LM.save_pretrained(cfg.save_dir,finetune=True,strategy=cfg.strategy)

    held_out_pyd = PytorchDataset(cfg.data_config, split="held_out")
    held_out_dataloader = torch.utils.data.DataLoader(
        held_out_pyd,
        batch_size=optimization_config.validation_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=held_out_pyd.collate,
        shuffle=False,
    )
    tuning_metrics = trainer.validate(model=LM, dataloaders=tuning_dataloader, ckpt_path="best")
    held_out_metrics = trainer.test(model=LM, dataloaders=held_out_dataloader, ckpt_path="best")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("Saving final metrics...")

        with open(cfg.save_dir / "tuning_metrics.json", mode="w") as f:
            json.dump(tuning_metrics, f)
        with open(cfg.save_dir / "held_out_metrics.json", mode="w") as f:
            json.dump(held_out_metrics, f)

    return tuning_metrics[0]["tuning_loss"], tuning_metrics, held_out_metrics
