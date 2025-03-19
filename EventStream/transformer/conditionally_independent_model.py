"""The conditionally independent event stream GPT model."""
from typing import Any

import torch
import wandb
import numpy as np

from ..data.types import DataModality, PytorchBatch
from .config import StructuredEventProcessingMode, StructuredTransformerConfig
from .generation.generation_utils import StructuredGenerationMixin
from .model_output import (
    GenerativeOutputLayerBase,
    GenerativeSequenceModelLabels,
    GenerativeSequenceModelLosses,
    GenerativeSequenceModelOutput,
    GenerativeSequenceModelPredictions,
)
from .transformer import (
    ConditionallyIndependentPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
    expand_mask,
    time_from_deltas,
)
from torchmetrics.classification import BinaryAUROC

class ConditionallyIndependentGenerativeOutputLayer(GenerativeOutputLayerBase):
    """The output layer for the conditionally independent event stream model.

    TODO(mmcdermott):
        Allow for use of NLL-beta throughout? https://github.com/mmcdermott/EventStreamGPT/issues/26

    Args:
        config: The overall model configuration.

    Raises:
        ValueError: If the model configuration does not indicate conditionally independent mode.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)
        if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
            raise ValueError(f"{config.structured_event_processing_mode} invalid!")
        

    def proxy_task(
            self,
            batch,
            classification_out,
            regression_out,
            TTE_dist,
            task: str = None,
            encoded=None,
            is_generation=None
            ):

        event_mask = batch["event_mask"]
        device = event_mask.device
    
        task_loss = None
        accuracy = None
        auroc_score = None

        match task:
            case "interruption_in_seq":
                pred_distributions = TTE_dist
                pred_time_deltas = pred_distributions.sample()
                
                filtered_time_deltas = pred_time_deltas.masked_fill(event_mask == False, 0)     # [:-1]??
                gen_times = filtered_time_deltas.cumsum(dim=1)
                is_within_7d = gen_times < (7*24*60)

                pred_event_types_logits = classification_out[1]["event_type"][1].logits  # (batch_size, seq_len, num_classes)
                pred_event_types_probs = torch.softmax(pred_event_types_logits, dim=2)  # (batch_size, seq_len, num_classes)

                interruption_index = self.config.event_types_idxmap["interruption"]
                is_interruption_prob = pred_event_types_probs[:, :, interruption_index]  # (batch_size, seq_len)

                any_interruption_in_7d_prob = (is_interruption_prob * is_within_7d).sum(dim=1)

                # pos_weight = (labels == 0).float().mean() / (0.000001+(labels == 1)).float().mean()

                loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=pos_weight)
                task_loss = loss_fn(any_interruption_in_7d_prob, batch.stream_labels["label"].float())

                pred_task_labels_binary = (any_interruption_in_7d_prob > 0.5).float()
                accuracy = (pred_task_labels_binary == batch.stream_labels["label"].float()).float().mean()


                auroc = BinaryAUROC()
                auroc_score = auroc(any_interruption_in_7d_prob, batch.stream_labels["label"].float())

            case "majority_class":
                # "DOES NOT WORK, BECAUSE MODE OPERATION IS NOT DIFFERENTIABLE!"
                event_types = classification_out[2]["event_type"].to(device)
                majority_classes, _ = torch.mode(event_types, dim=1)
                majority_classes = majority_classes.to(device)

                pred_event_labels_logits = classification_out[1]["event_type"][1].logits
                pred_event_labels_probs = torch.softmax(pred_event_labels_logits, dim=2)
                pred_event_labels = torch.argmax(pred_event_labels_probs, dim=2)
                pred_majority_classes, _ = torch.mode(pred_event_labels, dim=1)
                pred_majority_classes = pred_majority_classes.to(device)

                loss_fn = torch.nn.CrossEntropyLoss()
                task_loss = loss_fn(pred_majority_classes.float(), majority_classes.float())

                accuracy = (pred_majority_classes == majority_classes).float().mean()

                auroc = BinaryAUROC()
                auroc_score = auroc(pred_majority_classes.float(), majority_classes.float())


            case "class_distribution":
                event_types = classification_out[2]["event_type"].to(device)
                num_classes = classification_out[1]["event_type"][1].logits.shape[-1]
                batch_size, seq_len = event_types.size()

                target_distributions = torch.zeros(batch_size, num_classes).to(event_types.device)
                for i in range(batch_size):
                    for j in range(seq_len):
                        target_distributions[i, event_types[i, j]] += 1
                target_distributions = target_distributions / seq_len

                pred_event_labels_logits = classification_out[1]["event_type"][1].logits
                pred_event_labels_probs = torch.softmax(pred_event_labels_logits, dim=2).mean(dim=1) # Average probabilities across sequence

                task_loss = torch.mean((pred_event_labels_probs - target_distributions)**2)

                pred_argmax = torch.argmax(pred_event_labels_probs, dim=1)
                target_argmax = torch.argmax(target_distributions, dim=1)

                auroc = BinaryAUROC()
                specific_class_index = 0
                auroc_score = auroc(pred_event_labels_probs[:, specific_class_index], target_distributions[:, specific_class_index])

                print("task loss: ", task_loss)

            case "interruption_next_week_cls":

                pred_distributions = TTE_dist
                pred_time_deltas = pred_distributions.sample()
                
                filtered_time_deltas = pred_time_deltas.masked_fill(event_mask == False, 0)     # [:-1]??
                gen_times = filtered_time_deltas.cumsum(dim=1)
                is_next_week = gen_times > (7*24*60)

                pred_event_types_logits = classification_out[1]["event_type"][1].logits  # (batch_size, seq_len, num_classes)           # 0: losses, 1: predictions, 2: labels
                pred_event_types_probs = torch.softmax(pred_event_types_logits, dim=2)  # (batch_size, seq_len, num_classes)

                interruption_index = self.config.event_types_idxmap["interruption"]
                is_interruption_prob = pred_event_types_probs[:, :, interruption_index]  # (batch_size, seq_len)
                interruption_next_week_prob = 1 - torch.prod(1-is_interruption_prob, dim=1) # shape batch_size


                # create labels
                combined_mask = (event_mask & is_next_week)
                event_type_labels = classification_out[2]["event_type"].masked_fill(combined_mask==False,-2) + 1               # shift to get correct event type indices, same ones that are in self.config.event_type_idxmap
                interruption_next_week_label = (event_type_labels == interruption_index).any(dim=1).float()

                loss_fn = torch.nn.BCELoss()
                task_loss = loss_fn(interruption_next_week_prob, interruption_next_week_label)

                binary_interruption_pred = (interruption_next_week_prob > 0.5).float()
                accuracy = (binary_interruption_pred == interruption_next_week_label).float().mean()

                auroc = BinaryAUROC()
                auroc_score = auroc(interruption_next_week_prob, interruption_next_week_label)

                # hur använder vi endast första veckan i batchen för att göra predictions, så som det är nu är det nästan samma som interruption in seq fast kollar bara på andra veckan?


            case "time_to_next_interruption":
                # num_new_events = 1000

                # gen_times, gen_event_labels, gen_interruption = self.autoregressive_generate(batch=batch, encoded=encoded, num_new_events=num_new_events)
                # interruption_next_week_prob = 1 - torch.prod(1-gen_interruption, dim=1)    # calculate the probability of an interruption happening next week, based            
                # stream_labels = batch.stream_labels["label"].float()

                # Maccky style
                time_delta = batch.time_delta
                cumulative_time_delta = time_delta.cumsum(dim=1)
                max_true_index = batch.event_mask.cumsum(dim=1).argmax(dim=1)
                cumulative_time_delta_filtered = cumulative_time_delta[:, max_true_index]

                in_one_week = (cumulative_time_delta_filtered < 60*24*7).to(device)
                one_week_idx = (in_one_week.cumsum(dim=1).argmax(dim=1)).long()



                # get label for TTI task
                whole_event_encoded = encoded
                classification_measurements = set(self.classification_mode_per_measurement.keys())
                for_event_contents_prediction = torch.cat(
                (
                    torch.zeros_like(whole_event_encoded[:, 0, :]).unsqueeze(1),
                    whole_event_encoded[:, :-1, :],
                ),
                dim=1,
                )
                classification_out = self.get_classification_outputs(batch, for_event_contents_prediction, classification_measurements)         # 0: losses, 1: predictions, 2: labels
                event_type_labels = classification_out[2]["event_type"].to(torch.int64)                                                         # shape: batch_size x seq_len
                interruption_event_type_idx = self.config.event_types_idxmap["interruption"]                                                    # get event type index of interruption
                interruption_event_type_idx = torch.tensor(interruption_event_type_idx, dtype=torch.int64, device=event_type_labels.device)

                # find index in each sequence where the event type is interruption
                interruptions_in_seq = (event_type_labels == interruption_event_type_idx)
                interruption_indices = torch.nonzero(interruptions_in_seq, as_tuple=False)[:,0]
    
                tti_labels = torch.zeros((batch.batch_size,1))
                for i in range(batch.batch_size):
                    if len(interruption_indices) != 0:
                        
                        index = torch.searchsorted(interruption_indices, one_week_idx, right=True)    
                        tti_labels[i] = cumulative_time_delta_filtered[index]    

                    else:
                        tti_labels[i] = -1                                             # no interruption next week


                print(tti_labels)
                raise


            case _:
                raise ValueError(f"Task: {task} not implemented!")

        return task_loss, accuracy, auroc_score


    def forward(
        self,
        batch: PytorchBatch,
        encoded: torch.FloatTensor,
        is_generation: bool = False
    ) -> GenerativeSequenceModelOutput:
        """Returns the overall model output for the input batch.

    #     It takes the final hidden states from the encoder and runs them through various output layers to
    #     predict subsequent event timing and contents. It's difference from a nested attention variant is
    #     largely in that it predicts everything simultaneously.

    #     Args:
    #         batch: The batch of data to process.
    #         encoded: The encoded representation of the input data.
    #         is_generation: Whether or not we are in generation mode. If so, the output predictions are for the
    #             next event for both time and event contents; if not, then we shift the event contents
    #             predictoin back by one event in order to align with the labels.
    #     """
        # These are the containers we'll use to process the outputs
        classification_dists_by_measurement = {}
        classification_losses_by_measurement = None if is_generation else {}
        classification_labels_by_measurement = None if is_generation else {}
        regression_dists = {}
        regression_loss_values = None if is_generation else {}
        regression_labels = None if is_generation else {}
        regression_indices = None if is_generation else {}

        classification_measurements = set(self.classification_mode_per_measurement.keys())
        regression_measurements = set(
            self.config.measurements_for(DataModality.MULTIVARIATE_REGRESSION)
            + self.config.measurements_for(DataModality.UNIVARIATE_REGRESSION)
        )

        # encoded is of shape: (batch size, sequence length, config.hidden_size)
        bsz, seq_len, _ = encoded.shape
        whole_event_encoded = encoded

        # In this case, the whole_event_encoded representation actually is used to predict the next event's
        # contents, so it is what we want if we are in generative mode, but if we are not in generative mode
        # then to make it align with the labels we need to shift it to be in the right form. In particular, we
        # prepend a vector of zeros to be used to predict the contents of the first event (excluding the TTE
        # of the first event which is guaranteed to be zero) and we _don't_ predict the contents of the event
        # after the end of this sequence (as we have no way to judge them).

        if is_generation:
            for_event_contents_prediction = whole_event_encoded
        else:
            for_event_contents_prediction = torch.cat(
                (
                    torch.zeros_like(whole_event_encoded[:, 0, :]).unsqueeze(1),
                    whole_event_encoded[:, :-1, :],
                ),
                dim=1,
            )

        classification_out = self.get_classification_outputs(
            batch,
            for_event_contents_prediction,
            classification_measurements,
        )


        classification_dists_by_measurement.update(classification_out[1])
        if not is_generation:
            classification_losses_by_measurement.update(classification_out[0])
            classification_labels_by_measurement.update(classification_out[2])

        regression_out = self.get_regression_outputs(
            batch,
            for_event_contents_prediction,
            regression_measurements,
            is_generation=is_generation,
        )

        task_loss, accuracy, auroc_score = self.get_task_outputs(
            batch,
            for_event_contents_prediction,
            classification_out = classification_out
        )



        regression_dists.update(regression_out[1])
        if not is_generation:
            regression_loss_values.update(regression_out[0])
            regression_labels.update(regression_out[2])
            regression_indices.update(regression_out[3])

        TTE_LL_overall, TTE_dist, TTE_true = self.get_TTE_outputs(
            batch,
            whole_event_encoded,
            is_generation=is_generation,
        )

        # if batch.stream_labels:
        #     task_loss, accuracy, auroc_score = self.proxy_task(batch, classification_out, regression_out, TTE_dist, task = "interruption_next_week_cls", encoded=encoded)
        # else:
        #     task_loss = 0
        #     accuracy = None
        #     auroc_score = None
        
        
        return GenerativeSequenceModelOutput(
            **{
                "loss": (
                    task_loss
                    + sum(classification_losses_by_measurement.values())
                    + sum(regression_loss_values.values())
                    - TTE_LL_overall
                )
                if not is_generation
                else None,
                "losses": GenerativeSequenceModelLosses(
                    **{
                        "classification": classification_losses_by_measurement,
                        "regression": regression_loss_values,
                        "time_to_event": None if is_generation else -TTE_LL_overall,
                        "task_loss": task_loss,
                        "task_accuracy": accuracy,
                        "task_AUROC": auroc_score
                    }
                ),
                "preds": GenerativeSequenceModelPredictions(
                    classification=classification_dists_by_measurement,
                    regression=regression_dists,
                    regression_indices=regression_indices,
                    time_to_event=TTE_dist,
                ),
                "labels": GenerativeSequenceModelLabels(
                    classification=classification_labels_by_measurement,
                    regression=regression_labels,
                    regression_indices=regression_indices,
                    time_to_event=None if is_generation else TTE_true,
                ),
                "event_mask": batch["event_mask"],
                "dynamic_values_mask": batch["dynamic_values_mask"],
            }
        )







        # else:
        #     return GenerativeSequenceModelOutput(
        #     **{
        #         "loss": (
        #             sum(classification_losses_by_measurement.values())
        #             + sum(regression_loss_values.values())
        #             - TTE_LL_overall
        #         )
        #         if not is_generation
        #         else None,
        #         "losses": GenerativeSequenceModelLosses(
        #             **{
        #                 "classification": classification_losses_by_measurement,
        #                 "regression": regression_loss_values,
        #                 "time_to_event": None if is_generation else -TTE_LL_overall,
        #             }
        #         ),
        #         "preds": GenerativeSequenceModelPredictions(
        #             classification=classification_dists_by_measurement,
        #             regression=regression_dists,
        #             regression_indices=regression_indices,
        #             time_to_event=TTE_dist,
        #         ),
        #         "labels": GenerativeSequenceModelLabels(
        #             classification=classification_labels_by_measurement,
        #             regression=regression_labels,
        #             regression_indices=regression_indices,
        #             time_to_event=None if is_generation else TTE_true,
        #         ),
        #         "event_mask": batch["event_mask"],
        #         "dynamic_values_mask": batch["dynamic_values_mask"],
        #     }
        # )



class CIPPTForGenerativeSequenceModeling(StructuredGenerationMixin, StructuredTransformerPreTrainedModel):
    """The end-to-end model for conditionally independent generative sequence modelling.

    This model is a subclass of :class:`~transformers.StructuredTransformerPreTrainedModel` and is designed
    for generative pre-training over "event-stream" data, with inputs in the form of `PytorchBatch` objects.
    It is trained to solve the generative, multivariate, masked temporal point process problem over the
    defined measurements in the input data.

    This model largely simply passes the input data through a
    `ConditionallyIndependentPointProcessTransformer` followed by a
    `ConditionallyIndependentGenerativeOutputLayer`.

    Args:
        config: The overall model configuration.

    Raises:
        ValueError: If the model configuration does not indicate conditionally independent mode.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)

        if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
            raise ValueError(f"{config.structured_event_processing_mode} invalid!")

        self.encoder = ConditionallyIndependentPointProcessTransformer(config)
        self.output_layer = ConditionallyIndependentGenerativeOutputLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self, batch: PytorchBatch, past: tuple | None = None, **kwargs
    ) -> dict[str, Any]:
        """Returns model keyword arguments that have been modified for generation purposes.

        Args:
            batch: The batch of data to be transformed.
            past: The past state of the model, if any. If specified, it must be a tuple containing the past
                values over prior layers and heads.

            **kwargs: Additional keyword arguments. If "use_cache" is set in the kwargs to False, then the
                past state is ignored. If not, then the past state is passed through the model to accelerate
                generation, if past is not None then the batch is trimmed to the last element in the sequence,
                and the sequential attention mask is pre-computed.

        Raises:
            ValueError: If the past state is malformed or if there is a dep_graph_el_generation_target in the
                kwargs that is not None.
        """
        # only last sequence element in the batch if past is defined in kwargs
        batch.time = time_from_deltas(batch)

        use_cache = kwargs.get("use_cache", False)
        if not use_cache:
            return {**kwargs, "batch": batch}

        seq_attention_mask = expand_mask(batch.event_mask, batch.time_delta.dtype)

        dep_graph_el_generation_target = kwargs.get("dep_graph_el_generation_target", None)
        if dep_graph_el_generation_target is not None:
            raise ValueError(
                f"Can't use dep_graph_el_generation_target ({dep_graph_el_generation_target}) "
                "in a conditionally independent model."
            )

        match past:
            case None:
                pass

            case tuple():
                batch = batch.last_sequence_element_unsqueezed()

            case _:
                raise ValueError(f"{past} malformed!")

        return {
            **kwargs,
            "seq_attention_mask": seq_attention_mask,
            "batch": batch,
            "past": past,
        }

    def forward(
        self, batch: PytorchBatch, is_generation: bool = False, **kwargs
    ) -> GenerativeSequenceModelOutput:
        """This runs the full forward pass of the model.

        Args:
            batch: The batch of data to be transformed.
            is_generation: Whether or not the model is being used for generation.
            **kwargs: Additional keyword arguments, which are used for output structuring and are forwarded to
                the encoder. The model specifically looks for use_cache, output_attentions, and
                output_hidden_states keyword arguments, which control whether additional properties should be
                added to the output.

        Returns:
            The output of the model, which is a `GenerativeSequenceModelOutput` object.
        """
        use_cache = kwargs.get("use_cache", False)
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)

        encoded = self.encoder(batch, **kwargs)
        output = self.output_layer(batch, encoded.last_hidden_state, is_generation=is_generation)

        if use_cache:
            output["past_key_values"] = encoded.past_key_values

        if output_attentions:
            output["attentions"] = encoded.attentions

        if output_hidden_states:
            output["hidden_states"] = encoded.hidden_states

        return output
