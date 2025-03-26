from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from EventStream.data.dataset_polars import Dataset
from EventStream.evaluation.general_generative_evaluation import ESTForTrajectoryGeneration
from EventStream.data.pytorch_dataset import PytorchDataset
from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.transformer.lightning_modules.generative_modeling import PretrainConfig, ESTForGenerativeSequenceModelingLM
from EventStream.transformer.conditionally_independent_model import CIPPTForGenerativeSequenceModeling 
from EventStream.transformer.lightning_modules.fine_tuning_dev import FinetuneConfig
from EventStream.transformer.model_output import get_event_types

import torch
import hydra
import numpy as np

import matplotlib.pyplot as plt

def masked_idx_in_set(
    indices_T: torch.LongTensor, indices_set: set[int], mask: torch.BoolTensor
) -> torch.BoolTensor:
    return torch.where(
        mask, torch.any(torch.stack([(indices_T == i) for i in indices_set], 0), dim=0), False
    )

def get_shapes(nested_tuple, depth=0):
    """Recursively prints the shape of tensors inside a nested tuple."""
    if isinstance(nested_tuple, torch.Tensor):
        print("  " * depth + f"Tensor shape: {tuple(nested_tuple.shape)}")
    elif isinstance(nested_tuple, (tuple, list)):
        print("  " * depth + f"Tuple of length {len(nested_tuple)}:")
        for item in nested_tuple:
            get_shapes(item, depth + 1)
    else:
        print("  " * depth + f"Unknown type: {type(nested_tuple)}")



def inference(cfg):
    # Instantiate model
    model = CIPPTForGenerativeSequenceModeling(
        config=cfg.config
    )

    # Load pretrained weights
    model.from_pretrained(
        cfg.pretrained_weights_fp
    )

    held_out_pyd = PytorchDataset(config=cfg.data_config, split="held_out")    

    held_out_dataloader = DataLoader(
        held_out_pyd, batch_size=5, collate_fn=held_out_pyd.collate, shuffle=True
    )  
    
    mse = []

    for batch in held_out_dataloader:
        # OLD STUFF!!!
            # generated = model.generate(
            #     batch=batch,
            #     max_new_events=max_new_events,  # Note that this must be within the model's `max_seq_len` - the input data length
            #     do_sample=True,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     output_hidden_states=True
            # )
    

            # classification = generated.scores[0].classification
            # event_label_vocab = []
            # pred_logits = generated.scores[0].classification["event_label"][1].logits
            # pred_labels_idx = torch.argmax(pred_logits, dim=1)

        

            # gen_mask = generated.batch.event_mask[:, input_seq_len:]
            # gen_measurements = generated.batch.dynamic_measurement_indices[:, input_seq_len:, :]
            # gen_indices = generated.batch.dynamic_indices[:, input_seq_len:, :]
            # gen_time_deltas = generated.batch.time_delta[:, input_seq_len:]

            # gen_event_types = get_event_types(
            #     gen_measurements,
            #     gen_indices,
            #     cfg.config.measurements_idxmap["event_type"],
            #     cfg.config.vocab_offsets_by_measurement["event_type"],
            # )
        
            # #gen_event_types is of shape [batch_size, sequence_length]
            # # print(gen_event_types)
            # mask = torch.where(gen_event_types==1,1,0)
            # interruption = torch.argmax(mask,dim=1)

            # no_ones = (mask.sum(dim=1) == 0)  
            # interruption[no_ones] = -1
            # # print(interruption)
            # time_deltas_cum = gen_time_deltas.cumsum(dim=1)
            # # print(time_deltas_cum)
            # ttis = time_deltas_cum[torch.arange(len(interruption)),interruption].float()
            # target_ttis = batch.stream_labels['label'].float()

            # mse_score = torch.nn.functional.mse_loss(ttis,target_ttis)
            # # print(get_shapes(generated.hidden_states))
            # batch_size = generated.hidden_states[-1][-1].shape[0]
            # seq_len = generated.hidden_states[-1][-1].shape[1]
            # hidden_size = generated.hidden_states[-1][-1].shape[-1]
            # encoded = generated.hidden_states[-1][-1]#s.view(batch_size,seq_len*hidden_size)
            # print(encoded.shape)

            # if seq_len < cfg.config.max_seq_len:
            #         zeros_to_add = cfg.config.max_seq_len - seq_len
            #         padded = torch.zeros((encoded.shape[0], zeros_to_add, cfg.config.hidden_size))
            #         encoded = torch.cat([encoded, padded],dim=1)
            # print(encoded.shape)


        encoded = model.encoder(batch,
                                  use_cache = None,
                                  output_attentions=False,
                                  output_hidden_states=True,
                                  return_dict=True)
        
        
        last_hidden_state = encoded.last_hidden_state
        
        mse.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True).losses.TTI_mse.item())
    

       
        


    print(f'Mead TTI MSE evaluated on HELD OUT set: {np.mean(mse)}')
    # plot distribution

    # num_samples = 1000
    # batch_size = 2
   
    # x_values_1d = torch.linspace(0.01, 5, num_samples)
    # x_values_column = x_values_1d.unsqueeze(1)
    # input_tensor_for_logprob = x_values_column.repeat(1, batch_size)
    
    # # print(torch.exp(dist.log_prob(input_tensor_for_logprob)))

    # pdf_values = torch.exp(dist.log_prob(input_tensor_for_logprob))[:,0].squeeze().detach().numpy()

    # plt.figure(figsize=(8, 6))
    # plt.plot(x_values_1d.numpy(), pdf_values, label='Exponential PDF')
    # plt.xlabel('x (min)')
    # plt.ylabel('Probability Density)')
    # plt.title('Plot of Eneryield Exponential Distribution PDF')
    # plt.grid(True)
    # plt.legend()
    # plt.xlim(0, max(x_values_1d.numpy()))
    # plt.ylim(bottom=0)
    # plt.savefig('/home/filip-marcus/figures/eneryield_exp_pdf.png')


@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: FinetuneConfig):
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")


    return inference(cfg)



if __name__ == "__main__":
    main()