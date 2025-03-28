import dataclasses
import json
import os
from pathlib import Path
from typing import Any
import gc
import importlib.util
import wandb

from safetensors.torch import load_file

import lightning as L
import omegaconf
import torch
import torch.multiprocessing
import torchmetrics
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    BinaryAUROC
)
from transformers import get_polynomial_decay_schedule_with_warmup
from ..transformer import time_from_deltas, ConditionallyIndependentPointProcessInputLayer

from ...data.config import PytorchDatasetConfig
from ...data.pytorch_dataset import PytorchDataset
from ...data.types import DataModality, PytorchBatch
from ...data.data_embedding_layer import DataEmbeddingLayer
from ...utils import hydra_dataclass, task_wrapper
from ..conditionally_independent_model import CIPPTForGenerativeSequenceModeling
from ..config import (
    Averaging,
    MetricCategories,
    Metrics,
    MetricsConfig,
    OptimizationConfig,
    Split,
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from ..model_output import GenerativeSequenceModelOutput
from ..nested_attention_model import NAPPTForGenerativeSequenceModeling
from ..utils import expand_indexed_regression, str_summary
#from ..generation.generation_utils import StructuredGenerationMixin


#class ESTForGenerativeSequenceModelingLM(L.LightningModule,StructuredGenerationMixin):
class ESTForGenerativeSequenceModelingLM(L.LightningModule):
    """A PyTorch Lightning Module for a `ESTForGenerativeSequenceModeling`."""

    TRAIN_SKIP_METRICS = ("AUROC", "AUPRC", "per_class")
    CLASSIFICATION = {
        DataModality.SINGLE_LABEL_CLASSIFICATION,
        DataModality.MULTI_LABEL_CLASSIFICATION,
    }

    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        optimization_config: OptimizationConfig | dict[str, Any],
        metrics_config: MetricsConfig | dict[str, Any],
        pretrained_weights_fp: Path | None = None,
    ):
        """Initializes the Lightning Module.

        Args:
            config (`Union[StructuredEventstreamTransformerConfig, Dict[str, Any]]`):
                The configuration for the underlying
                `ESTForGenerativeSequenceModeling` model. Should be
                in the dedicated `StructuredTransformerConfig` class or be a dictionary
                parseable as such.
            optimization_config (`Union[OptimizationConfig, Dict[str, Any]]`):
                The configuration for the optimization process handled by the Lightning module. Should
                be in the dedicated `OptimizationConfig` class or be a dictionary parseable
                as such.
        """ 
        super().__init__()

        # If the configurations are dictionaries, convert them to class objects. They may be passed as
        # dictionaries when the lightning module is loaded from a checkpoint, so we need to support
        # this functionality.
        if type(config) is dict:
            config = StructuredTransformerConfig(**config)
        if type(optimization_config) is dict:
            optimization_config = OptimizationConfig(**optimization_config)
        if type(metrics_config) is dict:
            metrics_config = MetricsConfig(**metrics_config)

        self.config = config
        self.optimization_config = optimization_config
        self.metrics_config = metrics_config
        self.optimization_config.gradient_clip_val = 1
        self.optimization_config.gradient_accumulation = 1
        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "optimization_config": dataclasses.asdict(optimization_config),
            }
        )
        self.build_metrics()

        match config.structured_event_processing_mode:
            case StructuredEventProcessingMode.NESTED_ATTENTION:
                model_cls = NAPPTForGenerativeSequenceModeling
            case StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
                model_cls = CIPPTForGenerativeSequenceModeling
            case _:
                raise ValueError(
                    f"Unsupported structured event processing mode: {config.structured_event_processing_mode}"
                )
       

        if pretrained_weights_fp is None:
            self.model = model_cls(config)
        else:
            self.model = model_cls.from_pretrained(pretrained_weights_fp, config=config, ignore_mismatched_sizes=True)
           
    
            # ###### TEST 1 ######
            # original_config = self.model.config
            # new_config = original_config
            # new_config.vocab_sizes_by_measurement = {"event_type": 10, "dummy_static": 2, "event_label": 5}
            # new_config.vocab_offsets_by_measurement = {"event_type": 1, "dummy_static": 11, "event_idx": 13, "event_label": 14, "feature_0": 19, "feature_1": 20, "feature_10": 21, "feature_11": 22, "feature_12": 23, "feature_13": 24, "feature_14": 25, "feature_15": 26, "feature_16": 27, "feature_17": 28, "feature_18": 29, "feature_19": 30, "feature_2": 31, "feature_20": 32, "feature_21": 33, "feature_22": 34, "feature_23": 35, "feature_24": 36, "feature_25": 37, "feature_26": 38, "feature_27": 39, "feature_28": 40, "feature_29": 41, "feature_3": 42, "feature_30": 43, "feature_31": 44, "feature_32": 45, "feature_33": 46, "feature_34": 47, "feature_35": 48, "feature_36": 49, "feature_37": 50, "feature_38": 51, "feature_39": 52, "feature_4": 53, "feature_40": 54, "feature_41": 55, "feature_42": 56, "feature_43": 57, "feature_44": 58, "feature_45": 59, "feature_46": 60, "feature_47": 61, "feature_48": 62, "feature_5": 63, "feature_6": 64, "feature_7": 65, "feature_8": 66, "feature_9": 67}
            # new_config.measurements_idxmap = {"event_type": 1, "dummy_static": 2, "event_idx": 3, "event_label": 4, "feature_0": 5, "feature_1": 6, "feature_10": 7, "feature_11": 8, "feature_12": 9, "feature_13": 10, "feature_14": 11, "feature_15": 12, "feature_16": 13, "feature_17": 14, "feature_18": 15, "feature_19": 16, "feature_2": 17, "feature_20": 18, "feature_21": 19, "feature_22": 20, "feature_23": 21, "feature_24": 22, "feature_25": 23, "feature_26": 24, "feature_27": 25, "feature_28": 26, "feature_29": 27, "feature_3": 28, "feature_30": 29, "feature_31": 30, "feature_32": 31, "feature_33": 32, "feature_34": 33, "feature_35": 34, "feature_36": 35, "feature_37": 36, "feature_38": 37, "feature_39": 38, "feature_4": 39, "feature_40": 40, "feature_41": 41, "feature_42": 42, "feature_43": 43, "feature_44": 44, "feature_45": 45, "feature_46": 46, "feature_47": 47, "feature_48": 48, "feature_5": 49, "feature_6": 50, "feature_7": 51, "feature_8": 52, "feature_9": 53}
            # new_config.measurements_per_generative_mode = {"single_label_classification": ["event_type"], "multi_label_classification": ["event_label"], "univariate_regression": ["event_idx", "feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "feature_9", "feature_10", "feature_11", "feature_12", "feature_13", "feature_14", "feature_15", "feature_16", "feature_17", "feature_18", "feature_19", "feature_20", "feature_21", "feature_22", "feature_23", "feature_24", "feature_25", "feature_26", "feature_27", "feature_28", "feature_29", "feature_30", "feature_31", "feature_32", "feature_33", "feature_34", "feature_35", "feature_36", "feature_37", "feature_38", "feature_39", "feature_40", "feature_41", "feature_42", "feature_43", "feature_44", "feature_45", "feature_46", "feature_47", "feature_48"]}
            # new_config.event_types_idxmap = {"unbalance_i": 1, "harmonics_i": 2, "unbalance_u": 3, "current_deviation": 4, "VD": 5, "transient": 6, "interruption": 7, "harmonics_u": 8, "VD&current_deviation": 9, "current_deviation&unbalance_i": 10}
            # self.model.encoder.input_layer = ConditionallyIndependentPointProcessInputLayer(new_config)
            ###### TEST 2 ######
            # print('generative_modeling.py line 128: HÄR SÄTTS n_total_embeddings HARDCODAT!!! just nu 17, ändra beroende på dataset')
            self.model.encoder.input_layer.data_embedding_layer = DataEmbeddingLayer(n_total_embeddings=config.vocab_size, # Dataset/Task specific!!!,
            out_dim=config.hidden_size,
            categorical_embedding_dim=config.categorical_embedding_dim,
            numerical_embedding_dim=config.numerical_embedding_dim,
            static_embedding_mode=config.static_embedding_mode,
            split_by_measurement_indices=None,
            do_normalize_by_measurement_index=config.do_normalize_by_measurement_index,
            static_weight=config.static_embedding_weight,
            dynamic_weight=config.dynamic_embedding_weight,
            categorical_weight=config.categorical_embedding_weight,
            numerical_weight=config.numerical_embedding_weight,)
        

    def save_pretrained(self, model_dir: Path, finetune=False,strategy=None):
        if finetune:
            fp = model_dir / "finetune_weights"
        else:
            fp = model_dir / "pretrained_weights"
        print(fp)
        self.model.save_pretrained(fp)

    def build_metrics(self):
        """Build the various torchmetrics we'll use to track performance."""

        # For judging our ability to predict time-to-event, we'll use the following scores:
        #   1. Explained Variance
        #   2. Mean Squared Error
        #   3. Mean Squared Log Error
        self.tte_metrics = torch.nn.ModuleDict(
            {
                "MSE": torchmetrics.MeanSquaredError(),
                "MSLE": torchmetrics.MeanSquaredLogError(),
                "explained_variance": torchmetrics.ExplainedVariance(),
            }
        )

        self.metrics = torch.nn.ModuleDict()
        for task_type, measurements in self.config.measurements_per_generative_mode.items():
            for measurement in measurements:
                vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                if measurement not in self.metrics:
                    self.metrics[measurement] = torch.nn.ModuleDict()
                if task_type not in self.metrics[measurement]:
                    self.metrics[measurement][task_type] = torch.nn.ModuleDict()

                match task_type:
                    case DataModality.SINGLE_LABEL_CLASSIFICATION:
                        cat = MetricCategories.CLASSIFICATION
                        metrics = {
                            Metrics.ACCURACY: (
                                MulticlassAccuracy,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUROC: (
                                MulticlassAUROC,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                            Metrics.AUPRC: (
                                MulticlassAveragePrecision,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                        }
                        metric_kwargs = {
                            "num_classes": vocab_size,
                            "ignore_index": 0,
                            "validate_args": self.metrics_config.do_validate_args,
                        }
                    case DataModality.MULTI_LABEL_CLASSIFICATION:
                        cat = MetricCategories.CLASSIFICATION
                        metrics = {
                            Metrics.ACCURACY: (
                                MultilabelAccuracy,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUROC: (
                                MultilabelAUROC,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUPRC: (
                                MultilabelAveragePrecision,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                        }
                        metric_kwargs = {
                            "num_labels": vocab_size,
                            "validate_args": self.metrics_config.do_validate_args,
                        }
                    case DataModality.UNIVARIATE_REGRESSION:
                        cat = MetricCategories.REGRESSION
                        metrics = {
                            Metrics.MSE: (torchmetrics.MeanSquaredError, [None]),
                            Metrics.EXPLAINED_VARIANCE: (torchmetrics.ExplainedVariance, [None]),
                        }
                        metric_kwargs = {}
                    case DataModality.MULTIVARIATE_REGRESSION:
                        cat = MetricCategories.REGRESSION
                        metrics = {
                            Metrics.MSE: (torchmetrics.MeanSquaredError, [None]),
                            Metrics.EXPLAINED_VARIANCE: (
                                torchmetrics.ExplainedVariance,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                        }
                        metric_kwargs = {}
                    case _:
                        raise ValueError(f"Unrecognized modality {task_type}!")

                auc_kwargs = {
                    **metric_kwargs,
                    "thresholds": self.metrics_config.n_auc_thresholds,
                    "compute_on_cpu": True,
                }
                for metric, (metric_cls, averagings) in metrics.items():
                    if metric in (Metrics.AUROC, Metrics.AUPRC):
                        metric_cls_kwargs = {**auc_kwargs}
                    else:
                        metric_cls_kwargs = {**metric_kwargs}

                    for averaging in averagings:
                        if averaging is None:
                            metric_name = str(metric)
                            averaging_kwargs = {}
                        else:
                            metric_name = f"{averaging}_{metric}"
                            if metric == Metrics.EXPLAINED_VARIANCE:
                                if averaging == Averaging.MACRO:
                                    avg_str = "uniform_average"
                                elif averaging == Averaging.WEIGHTED:
                                    avg_str = "variance_weighted"
                                else:
                                    raise ValueError(f"{averaging} not supported for explained variance.")

                                averaging_kwargs = {"multioutput": avg_str}
                            else:
                                averaging_kwargs = {"average": averaging}

                        if self.metrics_config.do_log_any(cat, metric_name):
                            self.metrics[measurement][task_type][metric_name] = metric_cls(
                                **metric_cls_kwargs, **averaging_kwargs
                            )

    def _log_metric_dict(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metrics: dict[str, torchmetrics.Metric],
        split: Split,
        measurement: str,
        cat: MetricCategories,
    ):
        """This helper function logs the set of named metrics for the predictions `preds` and labels `labels`.

        Args:
            `preds` (`torch.Tensor`): The predictions for this metric calculation.
            `labels` (`torch.Tensor`): The labels for this metric calculation.
            `metrics` (`Dict[str, torchmetrics.Metric]`): The metrics to log, by name.
            `skip_metrics` (`Sequence[str]`):
                A list of metrics to skip. Entries are not full metric names, but rather are partial names and
                any metric whose name contains an element of `skip_metrics` will be skipped.
                For example, if `skip_metrics = ['AUROC', 'AUPRC']`, then a metric with name `'macro_AUROC'`
                or `'micro_AUPRC'` would be skipped, whereas a metric named `'weighted_accuracy'` would not.
            `split` (`str`): TODO
            `measurement` (`str`): The measurement of this metric calculation. Affects the log name.
        """

        for metric_name, metric in metrics.items():
            # We'll want to skip a metric if any element of our skip_metrics list is a substring of the metric
            # name:
            if not self.metrics_config.do_log(split, cat, metric_name):
                continue

            try:
                if split != Split.TRAIN:
                    # This is slightly more efficient if we only care about epoch-level outputs.
                    # Source: https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
                    metric.update(preds, labels)
                else:
                    metric(preds, labels)

                self.log(
                    f"{split}_{measurement}_{metric_name}",
                    metric,
                    batch_size=self.optimization_config.batch_size,
                    sync_dist=True,
                )
            except (ValueError, IndexError) as e:
                print(
                    f"Failed to compute {metric_name} for {measurement} "
                    f"with preds ({str_summary(preds)}) and labels ({str_summary(labels)}): {e}."
                )

    def log_tte_metrics(self, results: GenerativeSequenceModelOutput, split: Split):
        # The output of the model for time-to-event (and for regression targets as well) are pytorch
        # distribution objects, not scalars. So, for some evaluation metrics, we need to sample values from
        # those distributions to assess the metric.
        # TODO(mmd): We should likely be able to control how many samples are used, to minimize variance of
        # these results.
        tte_dist = results["preds"]["time_to_event"]
        tte_preds = tte_dist.sample()

        # After sampling, we also need to slice this down to just those intra-event-times that are actually
        # observed. This means we should drop the last sequence element (slice to `[:, :-1]` (as our observed
        # intra-event-times will only exist for the interior of our sequence), then further filter down to
        # just elements of the prediction for which the next sequence element was not masked
        # (mask via `results['event_mask'][:, 1:]`). We also need to filter the observed labels down to also
        # only be present for sequence elements where the next sequence element was truly observed.
        tte_preds = tte_preds[:, :-1][results["event_mask"][:, 1:]]
        tte_labels = results["labels"]["time_to_event"][results["event_mask"][:, 1:]]

        # Finally, we can log all relevant TTE metrics given these predictions and labels.
        self._log_metric_dict(
            preds=tte_preds,
            labels=tte_labels,
            metrics=self.tte_metrics,
            measurement="TTE",
            split=split,
            cat=MetricCategories.TTE,
        )

    def log_metrics(self, results: GenerativeSequenceModelOutput, split: Split):
        """Logs metric results for a given output result.

        Args:
            `results` (`transformerForGenerativeSequenceModelOutput`):
                The results to assess across the suite of metrics.
            `split` (`str`): The split that should be used when logging metric results.
        """
        # We always want to log the raw loss.
        log_kwargs = {"batch_size": self.optimization_config.batch_size, "sync_dist": True}
        self.log(f"{split}_loss", results["loss"], **log_kwargs)
    
        if self.metrics_config.do_log_only_loss(split):
            return
        # We start by logging the losses.
        if self.metrics_config.do_log(split, MetricCategories.LOSS_PARTS):
            self.log_dict(
                {f"{split}_{k}_cls_NLL": v for k, v in results["losses"]["classification"].items()},
                **log_kwargs,
            )
            self.log_dict(
                {f"{split}_{k}_reg_NLL": v for k, v in results["losses"]["regression"].items()},
                **log_kwargs,
            )
            self.log(f"{split}_TTE_reg_NLL", results["losses"]["time_to_event"], **log_kwargs)
        
        self.log("task_loss", results["losses"]["task_loss"],**log_kwargs)

        if results["losses"].task_accuracy:
            self.log("task_accuracy", results["losses"].task_accuracy, **log_kwargs)
  
        if results["losses"].task_AUROC:
            self.log("task_AUROC", results["losses"]["task_AUROC"], **log_kwargs)
        if results["losses"].task_loss:
            self.log("task_loss", results["losses"]["task_loss"],**log_kwargs)
        # Time-to-event
        if self.metrics_config.do_log(split, MetricCategories.TTE):
            self.log_tte_metrics(results, split)

        # Per data type
        for measurement, metrics_dict in self.metrics.items():
            mask = results["event_mask"]

            if not mask.any():
                continue

            for task_type, metrics in metrics_dict.items():
                if task_type in self.CLASSIFICATION and self.metrics_config.do_log(
                    split, MetricCategories.CLASSIFICATION
                ):
                    # For now, we ignore the is_observed distribution (the first element of the below tuple).
                    _, sample_dist = results["preds"]["classification"][measurement]
                    preds = sample_dist.logits
                    labels = results["labels"]["classification"][measurement]

                    # We need to filter these down to just those corresponding to observed events. Note that
                    # unlike TTE, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.

                    preds = preds[mask]
                    labels = labels[mask].long()

                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.CLASSIFICATION,
                    )

                elif task_type == DataModality.MULTIVARIATE_REGRESSION and self.metrics_config.do_log(
                    split, MetricCategories.REGRESSION
                ):
                    vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    _, dist = results["preds"]["regression"][measurement]
                    preds = dist.sample()[mask]
                    labels = results["labels"]["regression"][measurement][mask]

                    # However, as our regression output is actually indexed only to the group keys that are
                    # actually measured (tracked in `results['preds']['regression_indices']`, we need to
                    # expand our predictions and labels to be in the full vocabulary space for the metrics to
                    # work naturally.
                    preds_indices = results["preds"]["regression_indices"][measurement][mask]
                    labels_indices = results["labels"]["regression_indices"][measurement][mask]

                    # We also need to reflect just those data elements for which values were observed:
                    data_el_mask = results["dynamic_values_mask"][mask]

                    preds = preds[data_el_mask]
                    labels = labels[data_el_mask]
                    preds_indices = preds_indices[data_el_mask]
                    labels_indices = labels_indices[data_el_mask]

                    preds_expanded = expand_indexed_regression(preds, preds_indices, vocab_size)
                    labels_expanded = expand_indexed_regression(labels, labels_indices, vocab_size)

                    self._log_metric_dict(
                        preds=preds_expanded,
                        labels=labels_expanded,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.REGRESSION,
                    )
                elif task_type == DataModality.UNIVARIATE_REGRESSION and self.metrics_config.do_log(
                    split, MetricCategories.REGRESSION
                ):
                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    # We ignore the is observed distribution here.
                    _, dist = results["preds"]["regression"][measurement]
                    preds = dist.sample()[mask]
                    labels = results["labels"]["regression"][measurement][mask]

                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.REGRESSION,
                    )



    def training_step(self, batch: PytorchBatch, batch_idx: int) -> torch.Tensor:
        """Training step.

        Skips logging all AUROC, AUPRC, and per_class metric to save compute.
        """
        # if True:
        #     device = batch.device
        #     task_loss = 0
        #     interruption_idx = self.config.event_types_idxmap["interruption"]

        #     all_pred_types = []
        #     all_pred_values = []
        #     all_pred_times = []
        #     interruption_probs = []
        #     cumulative_time = torch.zeros((batch.batch_size, 1))

        #     event_mask = batch.event_mask.clone()
        #     time_delta = batch.time_delta.clone()
        #     dynamic_indices = batch.dynamic_indices.clone()
        #     dynamic_values = batch.dynamic_values.clone()
        #     # print("indices: ", dynamic_indices)
        #     # print("values ", dynamic_values)
        #     print(self.model(batch).labels[0].keys())
        #     raise
        #     i = 0
        #     while torch.mean(cumulative_time) < 24*60*7:
        #         print(i)
        #         i=i+1

        #         # create dynamic indices and dynamic values
                


        #         pred = self.model(batch)            # batch_size x seq_len
        #         event_type_pred = pred.preds.classification["event_type"][1].logits           # batch_size x seq_len x num_classes
                
        #         pred_values = []

        #         for key, (presence, normal_dist) in pred.preds.regression.items():
        #             sampled_value = normal_dist.rsample()
        #             pred_values.append(sampled_value)

        #         pred_values = torch.cat(pred_values, dim=-1)[:,-1,:]
        #         pred_time = pred.preds.time_to_event.rsample().squeeze(-1)[:,-1]
        #         next_event_type = event_type_pred.argmax(dim=-1)

        #         all_pred_types.append(next_event_type)
        #         all_pred_values.append(pred_values)
        #         all_pred_times.append(pred_time)

        #         # gör nya dynamic_values / dynamic_indices varje iteration


        #         print("dyn: ", dynamic_values.shape)
        #         print("predv alues: ", pred_values.shape)
        #         raise
        #         dynamic_values = torch.cat([dynamic_values[:, 1:, :], pred_values.unsqueeze(1)], dim=1)         # roll
        #         time_delta = torch.cat([time_delta[:, 1:], pred_time.unsqueeze(1)], dim=1)                      # roll

        #         # update batch
        #         batch.dynamic_indices = dynamic_indices
        #         batch.dynamic_values = dynamic_values
        #         batch.time_delta = time_delta


        #         # calculate predictions of cls interruption next week and TTI
        #         interruption_prob = event_type_pred[:,:,interruption_idx][:,-1]       # probability of each event being interruption
        #         interruption_probs.append(interruption_prob)                        # should be batch_size x 1



        #         cumulative_time = cumulative_time + time_delta

        #     interruption_next_week_prob = 1 - torch.prod(1-interruption_probs, dim=1)
        #     raise


        # labeler_fp = "data/processed/eneryield_event_type/task_dfs/task_df_eneryield_interruption_cls_one_week_ahead_labeler.py"
        # labeler_cls = import_class_from_file(labeler_fp, "TaskLabeler")
        # labeling_function = labeler_cls(config=self.config)
        
        # empirical_labels, labels_unpredicted = labeling_function(self.model.generate(
        #                                                                 batch,
                                                                    
        #                                                                 do_sample=True,
        #                                                                 return_dict_in_generate=False,
        #                                                                 output_scores=False,
        #                                                                 num_return_sequences=self.config.task_specific_params["num_samples"],
        #                                                                 output_attentions=False,
        #                                                                 output_hidden_states=False,
        #                                                                 use_cache=True,
        #                                                                 max_length=200
        #                                                             ),
        #                                                             input_seq_len=batch.sequence_length,
        #                                                         )
        
        
        # stream_labels_float = batch.stream_labels['label'].squeeze().float()
        
        # # print("empirical labels: ", type(torch.argmax(empirical_labels,dim=1)[1]))
        # # print("stream labels: ",type(stream_labels_float[1]))
        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss = loss_fn(torch.argmax(empirical_labels,dim=1).float(),stream_labels_float)
        # # print("Loss: ", loss)
        # accuracy = (torch.argmax(empirical_labels,dim=1).float() == stream_labels_float).float().mean()
        # auroc = BinaryAUROC()
        # auroc_score = auroc(torch.argmax(empirical_labels,dim=1).float(),stream_labels_float)
        # wandb.init()
        # wandb.log({'task accuracy': accuracy})
        # wandb.log({'task AUROC':auroc_score})
        



        # ########### calculate task loss ###########

        # out = self.model(batch)
        # event_mask = out.event_mask
        # device = event_mask.device


        # ########## get predicted time deltas ###########

        # pred_distributions = out.preds.time_to_event
        # pred_time_deltas = pred_distributions.sample()
        
        # filtered_time_deltas = pred_time_deltas.masked_fill(out.event_mask == False, 0)     # [:-1] # kan vara lite kaiko här
        # gen_times = filtered_time_deltas.cumsum(dim=1)
        # is_within_7d = gen_times < (7*24*60)


        # ############# get predicted event_types and correct labels ############

        # pred_event_types_logits = out.preds.classification["event_type"][1].logits
        # pred_event_types = torch.argmax(pred_event_types_logits,dim=2)             # shape batch_size x seq_len
        # pred_event_types = pred_event_types.masked_fill(event_mask == False, -1).to(device)    # set event_type to -1 where event_mask==False


        # pred_event_labels_logits = out.preds.classification["event_label"][1].logits
        # pred_event_labels = torch.argmax(pred_event_labels_logits,dim=2)
        # pred_event_labels = pred_event_labels.masked_fill(event_mask == False, -1).to(device)

        # ########## calculate boolean if interruption is predicted in the next 7 days ##############

        # interruption_index = self.config.event_types_idxmap["interruption"]
        # is_interruption = (pred_event_types == interruption_index)
        # any_interruption_in_7d = (is_interruption & is_within_7d).any(dim=1)


        # ######### calculate the binary cross-entropy loss ############

        # pred_task_labels = any_interruption_in_7d
        # stream_labels = batch.stream_labels["label"]

        # loss_fn = torch.nn.CrossEntropyLoss()
        # task_loss = loss_fn(pred_task_labels.float(), stream_labels.float())

        # accuracy = (pred_task_labels.float() == stream_labels.float()).float().mean()
        # auroc = BinaryAUROC()
        # auroc_score = auroc(pred_task_labels.float(), stream_labels.float())

        # ############ wandb log ############
        # wandb.init()
        # wandb.log({'task loss': task_loss,
        #            'task accuracy': accuracy,
        #             'task AUROC': auroc_score
        #            })
       
        

        out = self.model(batch)
        task_losses = out["losses"].task_loss
        auroc_score = out["losses"].task_AUROC
        accuracy = out["losses"].task_accuracy
        # wandb.init()
        # wandb.log({
        #         'task loss': task_losses,
        #         'task accuracy': accuracy,
        #         'task AUROC': auroc_score
        #         })
    

        self.log_metrics(out, split=Split.TRAIN)

        return out["loss"]

    def validation_step(self, batch: PytorchBatch, batch_idx: int):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """

        out = self.model(batch)
        self.log_metrics(out, split=Split.TUNING)

    def test_step(self, batch: PytorchBatch, batch_idx: int):
        """Validation step.

        Differs from training only in that it does not skip metrics.
        """
        out = self.model(batch)
        self.log_metrics(out, split=Split.HELD_OUT)

    def configure_optimizers(self):
        """Configures optimizer and learning rate scheduler.

        Currently this module uses the AdamW optimizer, with configurable weight_decay, with a learning rate
        warming up from 0 on a per-step manner to the configurable `self.optimization_config.init_lr`, then
        undergoes polynomial decay as specified via `self.optimization_config`.
        """
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimization_config.init_lr,
            weight_decay=self.optimization_config.weight_decay,
        )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=self.optimization_config.lr_num_warmup_steps,
            num_training_steps=self.optimization_config.max_training_steps,
            power=self.optimization_config.lr_decay_power,
            lr_end=self.optimization_config.end_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # def prepare_inputs_for_generation(
    #     self, batch: PytorchBatch, past: tuple | None = None, **kwargs
    #     ) -> dict[str, Any]:
    #     """Returns model keyword arguments that have been modified for generation purposes.

    #     Args:
    #         batch: The batch of data to be transformed.
    #         past: The past state of the model, if any. If specified, it must be a tuple containing the past
    #             values over prior layers and heads.

    #         **kwargs: Additional keyword arguments. If "use_cache" is set in the kwargs to False, then the
    #             past state is ignored. If not, then the past state is passed through the model to accelerate
    #             generation, if past is not None then the batch is trimmed to the last element in the sequence,
    #             and the sequential attention mask is pre-computed.

    #     Raises:
    #         ValueError: If the past state is malformed or if there is a dep_graph_el_generation_target in the
    #             kwargs that is not None.
    #     """
    #     # only last sequence element in the batch if past is defined in kwargs
    #     batch.time = time_from_deltas(batch)

    #     use_cache = kwargs.get("use_cache", False)
    #     if not use_cache:
    #         return {**kwargs, "batch": batch}

    #     seq_attention_mask = expand_mask(batch.event_mask, batch.time_delta.dtype)

    #     dep_graph_el_generation_target = kwargs.get("dep_graph_el_generation_target", None)
    #     if dep_graph_el_generation_target is not None:
    #         raise ValueError(
    #             f"Can't use dep_graph_el_generation_target ({dep_graph_el_generation_target}) "
    #             "in a conditionally independent model."
    #         )

    #     match past:
    #         case None:
    #             pass

    #         case tuple():
    #             batch = batch.last_sequence_element_unsqueezed()

    #         case _:
    #             raise ValueError(f"{past} malformed!")

    #     return {
    #         **kwargs,
    #         "seq_attention_mask": seq_attention_mask,
    #         "batch": batch,
    #         "past": past,
    #     }


def import_class_from_file(module_path, class_name):
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
SKIP_CFG_PARAMS = {"seq_attention_layers", "dep_graph_attention_layers", "hidden_size"}


@hydra_dataclass
class PretrainConfig:
    do_overwrite: bool = False
    seed: int = 1

    config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "_target_": "EventStream.transformer.config.StructuredTransformerConfig",
            **{
                k: v
                for k, v in StructuredTransformerConfig(measurements_per_dep_graph_level=[]).to_dict().items()
                if k not in SKIP_CFG_PARAMS
            },
            "cohort_name": None
        }
    )
    optimization_config: OptimizationConfig = dataclasses.field(default_factory=lambda: OptimizationConfig())
    data_config: PytorchDatasetConfig = dataclasses.field(default_factory=lambda: PytorchDatasetConfig())
    pretraining_metrics_config: MetricsConfig = dataclasses.field(
        default_factory=lambda: MetricsConfig(
            include_metrics={Split.TRAIN: {MetricCategories.LOSS_PARTS: True}},
        )
    )
    final_validation_metrics_config: MetricsConfig = dataclasses.field(
        default_factory=lambda: MetricsConfig(do_skip_all_metrics=False)
    )

    trainer_config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": "auto",
            "detect_anomaly": False,
            "default_root_dir": "${save_dir}/model_checkpoints",
            "log_every_n_steps": 10,
            "strategy": "ddp_find_unused_parameters_true"
            # "precision" : 16
        }
    )

    experiment_dir: str = omegaconf.MISSING
    
    # save_dir: str = "${experiment_dir}/pretrain/" + str(config.)
    save_dir: str = "${experiment_dir}/pretrain/"

    wandb_logger_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "generative_event_stream_transformer",
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

    do_final_validation_on_metrics: bool = True
    do_use_filesystem_sharing: bool = True

    # compile: bool = True

    def __post_init__(self):
        self.save_dir=self.save_dir + str(self.config.cohort_name)
        if type(self.save_dir) is str and self.save_dir != omegaconf.MISSING:
            self.save_dir = Path(self.save_dir)
        if "max_epochs" in self.trainer_config:
            raise ValueError("Max epochs is set in the optimization_config, not the trainer config!")
        if "callbacks" in self.trainer_config:
            raise ValueError("Callbacks are built internally, not set via trainer_config!")


@task_wrapper
def train(cfg: PretrainConfig):
    """Runs the end to end training procedure for the pre-training model.

    Args:
        cfg: The pre-training config defining the generative modeling task.
    """

    L.seed_everything(cfg.seed)
    if cfg.do_use_filesystem_sharing:
        torch.multiprocessing.set_sharing_strategy("file_system")
    # print(cfg.data_config)
    train_pyd = PytorchDataset(cfg.data_config, split="train")
    tuning_pyd = PytorchDataset(cfg.data_config, split="tuning")

    config = cfg.config
    optimization_config = cfg.optimization_config
    data_config = cfg.data_config

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
        cfg.pretraining_metrics_config.to_json_file(
            cfg.save_dir / "pretraining_metrics_config.json", do_overwrite=cfg.do_overwrite
        )
        cfg.final_validation_metrics_config.to_json_file(
            cfg.save_dir / "final_validation_metrics_config.json", do_overwrite=cfg.do_overwrite
        )
    # print(config)
    # Model
    LM = ESTForGenerativeSequenceModelingLM(
        config=config,
        optimization_config=optimization_config,
        metrics_config=cfg.pretraining_metrics_config,
    )

    # TODO(mmd): Get this working!
    # if cfg.compile:
    #     print("Compiling model!")
    #     LM = torch.compile(LM)

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
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if optimization_config.patience is not None:
        callbacks.append(
            EarlyStopping(monitor="tuning_loss", mode="min", patience=optimization_config.patience)
        )

    trainer_kwargs = dict(
        **cfg.trainer_config,
        max_epochs=optimization_config.max_epochs,
        callbacks=callbacks,
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

    # Fitting model
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model=LM, train_dataloaders=train_dataloader, val_dataloaders=tuning_dataloader)

    LM.save_pretrained(cfg.save_dir)

    if cfg.do_final_validation_on_metrics:
        held_out_pyd = PytorchDataset(cfg.data_config, split="held_out")
        held_out_dataloader = torch.utils.data.DataLoader(
            held_out_pyd,
            batch_size=optimization_config.validation_batch_size,
            num_workers=optimization_config.num_dataloader_workers,
            collate_fn=held_out_pyd.collate,
            shuffle=False,
        )

        LM.metrics_config = cfg.final_validation_metrics_config
        LM.build_metrics()

        tuning_metrics = trainer.validate(model=LM, dataloaders=tuning_dataloader)
        held_out_metrics = trainer.test(model=LM, dataloaders=held_out_dataloader)

        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("Saving final metrics...")

            with open(cfg.save_dir / "tuning_metrics.json", mode="w") as f:
                json.dump(tuning_metrics, f)
            with open(cfg.save_dir / "held_out_metrics.json", mode="w") as f:
                json.dump(held_out_metrics, f)

        return tuning_metrics[0]["tuning_loss"], tuning_metrics, held_out_metrics

    return None
