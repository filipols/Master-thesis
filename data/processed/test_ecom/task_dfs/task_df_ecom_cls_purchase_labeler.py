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


class TaskLabeler(Labeler):                                                             # input: batch och sequence_len, seq_len används för att avgöra var genererade datan börjar
    def __call__(
        self, batch: PytorchBatch, input_seq_len: int
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:


        gen_mask = batch.event_mask[:, input_seq_len:]                                  # en vektor som innehåller massa False, och ett random True
        gen_measurements = batch.dynamic_measurement_indices[:, input_seq_len:, :]      # tensor/matris med index?
        gen_indices = batch.dynamic_indices[:, input_seq_len:, :]

        print("Event types index map:", self.config.event_types_idxmap)

        gen_event_types = get_event_types(                                              # obtains the event types from the model output
            gen_measurements,
            gen_indices,
            self.config.measurements_idxmap["event_type"],
            self.config.vocab_offsets_by_measurement["event_type"],
        )

        # gen_event_types is of shape [batch_size, sequence_length]


        # print('gen event type shape',gen_event_types.shape)
        # print('batch size', batch.batch_size)
        # print('sequence_len: ', input_seq_len)

        event_types_labels = ['view', 'cart', 'purchase']

        event_type_sets = {
            event_name: {self.config.event_types_idxmap[event_name]} for event_name in event_types_labels
        }


        # Initialize predictions and unknown event tracking
        pred_labels = []
        unknown_pred = []

        for event_name, event_set in event_type_sets.items():
            event_mask = masked_idx_in_set(gen_event_types, event_set, gen_mask)

            # Generate prediction for the current event type
            pred_event = torch.where(
                event_mask,
                torch.ones_like(event_mask, dtype=torch.long),
                torch.zeros_like(event_mask, dtype=torch.long),
            )

            pred_labels.append(pred_event)
            unknown_pred.append(event_mask == 0)

        # Stack the event predictions for each event type
        pred_labels = torch.stack(pred_labels, dim=1)  # Shape: [batch_size, num_event_types, sequence_length]
        unknown_pred = torch.stack(unknown_pred, dim=1)  # Shape: [batch_size, num_event_types, sequence_length]
        
        # Combine to form the final event predictions (if necessary)
        unknown_pred = unknown_pred.all(dim=1)  # If all event types are unknown for a sequence, mark as unknown

        return pred_labels, unknown_pred                                            # unknown is for when the model is not sure what the correct label is?



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