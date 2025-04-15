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
from safetensors.torch import load_file
import torch
import hydra
import numpy as np

import matplotlib.pyplot as plt

import warnings
import torchmetrics

# Suppress all UserWarnings from torchmetrics
warnings.filterwarnings("ignore", module="torchmetrics")

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

def check_pretrained(model):
    for name, param in model.named_parameters():
        # Check if the weights are all zeros or initialized
        if param.abs().sum() > 0:  # Non-zero weight indicates initialization
            print(f"Layer {name} has non-zero weights")
        else:
            print(f"Layer {name} has zero weights")






def inference(cfg):

    print('----------------------------------------------')
    print(f'INFERENCE ON TASK: {cfg.task_type}')
    print('----------------------------------------------')
    # Instantiate model
    model = CIPPTForGenerativeSequenceModeling(
        config=cfg.config
    )
    

    initial_weights = {name: param.clone() for name, param in model.named_parameters()}

   
    
    # Load pretrained weights
    checkpoint = load_file(cfg.pretrained_weights_fp / 'model.safetensors')
    model.load_state_dict(checkpoint)


    # Compare parameters
    total_params = 0
    params_changed = 0
    params_unchanged = 0
    for name, param in model.named_parameters():
        total_params+=1
        if not torch.equal(initial_weights[name], param):
            # print(f"{name} has been updated with pretrained weights.")
            params_changed +=1
        else:
            params_unchanged+=1
            # print(f"{name} remains unchanged!")

    print(f"Total parameters: {total_params} | Updated params: {params_changed} | Unchanged params: {params_unchanged}")
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    held_out_pyd = PytorchDataset(config=cfg.data_config, split="held_out")    

    held_out_dataloader = DataLoader(
        held_out_pyd, batch_size=8, collate_fn=held_out_pyd.collate, shuffle=False
    )  

    if cfg.task_type == "event_label":
        # Metrics:
        # 1. Accuracy
        # 2. AUROC
        # 3. AP
        # Return:
        # 1. event_label_accuracy
        # 2. event_label_auroc
        # 3. event_label_ap
        
        accuracy = []
        auroc = []
        ap = []
        for batch in held_out_dataloader:
            encoded = model.encoder(batch,
                                        use_cache = None,
                                        output_attentions=False,
                                        output_hidden_states=True,
                                        return_dict=True)
            
            accuracy.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.task_accuracy)
            auroc.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.task_AUROC.item())
            ap.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.average_precision.item())
        print('-------------------------------------------------------------------')
        print('METRICS:')
        print('-------------------------------------------------------------------')
        print(f'Mean Accuracy evaluated on HELD OUT set: {np.mean(accuracy)}')
        print(f'Mean AUROC evaluated on HELD OUT set: {np.mean(auroc)}')
        print(f'Mean AP evaluated on HELD OUT set: {np.mean(ap)}')
        print('-------------------------------------------------------------------')
    elif cfg.task_type == "interruption":
        # Metrics:
        # 1. Accuracy
        # 2. AUROC
        # 3. AP
        # Return:
        # 1. interruption_accuracy
        # 2. interruption_auroc
        # 3. interruption_ap

        accuracy = []
        auroc = []
        ap = []
        for batch in held_out_dataloader:
            encoded = model.encoder(batch,
                                        use_cache = None,
                                        output_attentions=False,
                                        output_hidden_states=True,
                                        return_dict=True)
            
            accuracy.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.task_accuracy)
            auroc.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.task_AUROC.item())
            ap.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.average_precision.item())
        print('-------------------------------------------------------------------')
        print('METRICS:')
        print('-------------------------------------------------------------------')
        print(f'Mean Accuracy evaluated on HELD OUT set: {np.mean(accuracy)}')
        print(f'Mean AUROC evaluated on HELD OUT set: {np.mean(auroc)}')
        print(f'Mean AP evaluated on HELD OUT set: {np.mean(ap)}')
        print('-------------------------------------------------------------------')

    elif cfg.task_type == "class_dist":
        # Metrics:
        # 1. MSE
        # Return:
        # 1. class_dist_mse

        mse = []
        for batch in held_out_dataloader:
            encoded = model.encoder(batch,
                                        use_cache = None,
                                        output_attentions=False,
                                        output_hidden_states=True,
                                        return_dict=True)
            
            mse.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.mse)
        print('-------------------------------------------------------------------')
        print('METRICS:')
        print('-------------------------------------------------------------------')
        print(f'Mean MSE evaluated on HELD OUT set: {mse}')
        print('-------------------------------------------------------------------')
    elif cfg.task_type == "tti":
        # Metrics:
        # 1. MSE
        # Return:
        # 1. class_dist_mse
        mse = []
        for batch in held_out_dataloader:
            encoded = model.encoder(batch,
                                        use_cache = None,
                                        output_attentions=False,
                                        output_hidden_states=True,
                                        return_dict=True)

            mse.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.mse)
        print('-------------------------------------------------------------------')
        print('METRICS:')
        print('-------------------------------------------------------------------')
        print(f'Mean MSE evaluated on HELD OUT set: {np.mean(mse):.4f}')
        print('-------------------------------------------------------------------')
    

@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: FinetuneConfig):
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")


    return inference(cfg)



if __name__ == "__main__":
    main()