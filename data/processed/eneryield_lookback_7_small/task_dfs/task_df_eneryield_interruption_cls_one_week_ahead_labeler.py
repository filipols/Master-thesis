import torch
from EventStream.data.pytorch_dataset import PytorchBatch
from EventStream.transformer.model_output import get_event_types
from EventStream.transformer.zero_shot_labeler import Labeler


def masked_idx_in_set(
    indices_T: torch.LongTensor, indices_set: set[int], mask: torch.BoolTensor
) -> torch.BoolTensor:
    return torch.where(
        mask, torch.any(torch.stack([(indices_T == i) for i in indices_set], 0), dim=0), False
    )


class TaskLabeler(Labeler):
    def __call__(
        self, batch: PytorchBatch, input_seq_len: int
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        gen_mask = batch.event_mask[:, input_seq_len:]
        gen_measurements = batch.dynamic_measurement_indices[:, input_seq_len:, :]
        gen_indices = batch.dynamic_indices[:, input_seq_len:, :]

        gen_event_types = get_event_types(
            gen_measurements,
            gen_indices,
            self.config.measurements_idxmap["event_type"], #Change from event_type to event_label
            self.config.vocab_offsets_by_measurement["event_type"], #Change from event_type to event_label
        )
        # print('BOBBAFETT: ', self.config.event_types_idxmap)


        # Assuming 'interruption' is a single label, get its index
        interruption_index = self.config.event_types_idxmap.get("interruption") #Change from event_types_idxmap to event_labels_idxmap
        
        print('BADDABING; ', batch)


        if interruption_index is None:
            raise ValueError("Label 'interruption' not found in event_labels_idxmap.")

        gen_time_deltas = batch.time_delta[:, input_seq_len - 1 : -1]
        gen_times = gen_time_deltas.cumsum(dim=1)

        # Check if any interruption occurs within 7 days (7 * 24 * 60 minutes)
        is_within_7d = gen_times < (7 * 24 * 60)
        is_interruption = masked_idx_in_set(
            gen_event_types, {interruption_index}, gen_mask
        )

        any_interruption_within_7d = (is_interruption & is_within_7d).any(dim=1)

        # Create binary labels: [no_interruption, interruption]
        pred_no_interruption = ~any_interruption_within_7d
        pred_interruption = any_interruption_within_7d

        pred_labels = torch.stack([pred_no_interruption, pred_interruption], 1).float()

        # Unknown predictions are those where neither interruption nor no interruption is predicted.
        unknown_pred = (~pred_interruption) & (~pred_no_interruption)


        # print('TITTTAAAA: ', pred_labels)

        return pred_labels, unknown_pred



# import torch
# from EventStream.data.pytorch_dataset import PytorchBatch
# from EventStream.transformer.model_output import get_event_types
# from EventStream.transformer.zero_shot_labeler import Labeler


# def masked_idx_in_set(
#     indices_T: torch.LongTensor, indices_set: set[int], mask: torch.BoolTensor
# ) -> torch.BoolTensor:
#     return torch.where(
#         mask, torch.any(torch.stack([(indices_T == i) for i in indices_set], 0), dim=0), False
#     )


# class TaskLabeler(Labeler):
#     def __call__(
#         self, batch: PytorchBatch, input_seq_len: int
#     ) -> tuple[torch.LongTensor, torch.BoolTensor]:
#         gen_mask = batch.event_mask[:, input_seq_len:]
#         gen_measurements = batch.dynamic_measurement_indices[:, input_seq_len:, :]
#         gen_indices = batch.dynamic_indices[:, input_seq_len:, :]

#         gen_event_types = get_event_types(
#             gen_measurements,
#             gen_indices,
#             self.config.measurements_idxmap["event_type"],
#             self.config.vocab_offsets_by_measurement["event_type"],
#         )

#         # gen_event_types is of shape [batch_size, sequence_length]




#         # CHAT

#         event_types_labels = ['view', 'cart', 'purchase']

#         event_type_sets = {
#             event_name: {i for et, i in self.config.event_types_idxmap.items() if event_name in et.split("&")}
#             for event_name in event_types_labels  # Assuming this is a list of event names
#         }

#         print(event_type_sets)

#         # Initialize predictions and unknown event tracking
#         pred_labels = []
#         unknown_pred = []

#         for event_name, event_set in event_type_sets.items():
#             event_mask = masked_idx_in_set(gen_event_types, event_set, gen_mask)

#             # Generate prediction for the current event type
#             pred_event = torch.where(
#                 event_mask,
#                 torch.ones_like(event_mask, dtype=torch.long),
#                 torch.zeros_like(event_mask, dtype=torch.long),
#             )

#             pred_labels.append(pred_event)
#             unknown_pred.append(event_mask == 0)

#         # Stack the event predictions for each event type
#         pred_labels = torch.stack(pred_labels, dim=1)  # Shape: [batch_size, num_event_types, sequence_length]
#         unknown_pred = torch.stack(unknown_pred, dim=1)  # Shape: [batch_size, num_event_types, sequence_length]
        
#         # Combine to form the final event predictions (if necessary)
#         unknown_pred = unknown_pred.all(dim=1)  # If all event types are unknown for a sequence, mark as unknown

#         # print('pred labels: ',pred_labels)

#         return pred_labels, unknown_pred