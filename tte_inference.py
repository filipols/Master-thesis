from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from EventStream.data.dataset_polars import Dataset
from EventStream.evaluation.general_generative_evaluation import ESTForTrajectoryGeneration
from EventStream.data.pytorch_dataset import PytorchDataset
from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.transformer.lightning_modules.generative_modeling import PretrainConfig
from EventStream.transformer.conditionally_independent_model import CIPPTForGenerativeSequenceModeling 
from EventStream.transformer.lightning_modules.fine_tuning import FinetuneConfig

import torch
import hydra

import matplotlib.pyplot as plt

def inference(cfg):

    model = CIPPTForGenerativeSequenceModeling(
        config=cfg.config
    )

    held_out_pyd = PytorchDataset(config=cfg.data_config, split="train")    

    sample_dataloader = DataLoader(
        held_out_pyd, batch_size=1, collate_fn=held_out_pyd.collate, shuffle=False
    )

    next(iter(sample_dataloader))
    sample_batch = next(iter(sample_dataloader))    

    # new_time_delta = torch.tensor([[4,2]])
    # sample_batch.time_delta = new_time_delta
    # print('sample_batch: ', sample_batch.time_delta)

    max_new_events=1
    generated = model.generate(
        sample_batch,
        max_new_events=max_new_events,  # Note that this must be within the model's `max_seq_len` - the input data length
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=False,
    )
    test = model(sample_batch)
    dist = test['preds']['time_to_event']

    print(test)

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



    # old stuff


    # scores = generated.scores
    # print('Sample: ', scores.time_to_event.sample())
    # classification = generated.scores[0].classification

    # print('TTE: ', generated)

    # event_labels_vocab = ['interruption', 'VD', 'unbalance_u', 'transient', 'interruption end', 'normal', 'current_deviation', 'unbalance_i', 'ongoing interruption', 'harmonics_i', 'harmonics_u']

    # print('Predicted event label:', generated.scores)
    # predicted_logits = generated.scores.classification['event_label'][1].logits  # Extract logits
    # predicted_labels_idx = torch.argmax(predicted_logits, dim=1)  # Get the index of the max logit


    # for i in range(len(predicted_labels_idx)):
    #     predicted_label = event_labels_vocab[i]
    #     print(f"Predicted TTE for batch {i}: ", generated.batch.time_delta[i,:])
    #     print('Ground truth TTE: ', generated.batch.stream_labels['label'])
    #     print(f"Predicted event label for batch {i}: {predicted_label}")


    # print(generated)


@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: FinetuneConfig):
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")


    return inference(cfg)



if __name__ == "__main__":
    main()