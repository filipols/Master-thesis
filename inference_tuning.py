from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torchmetrics.classification
from sklearn.isotonic import IsotonicRegression
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


def fit_isotonic_regression(calib_loader, model, device="cpu"):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in calib_loader:
            encoded = model.encoder(batch, output_hidden_states=False, return_dict=True)
            out_tuple = model.output_layer(batch, encoded.last_hidden_state, is_generation=True)
            preds = out_tuple[5].detach().cpu()
            labels = out_tuple[6].detach().cpu()
            all_probs.append(preds)
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).view(-1).numpy()
    all_labels = torch.cat(all_labels).view(-1).numpy()

    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(all_probs, all_labels)

    return iso_reg




def inference(cfg):

    print('----------------------------------------------')
    print(f'INFERENCE ON TASK: {cfg.task_type}')
    print('----------------------------------------------')
    # Instantiate model
    model = CIPPTForGenerativeSequenceModeling(
        config=cfg.config
    )
    

    initial_weights = {name: param.clone() for name, param in model.named_parameters()}

   
    threshold_ = cfg.threshold
    # Load pretrained weights
    checkpoint = load_file(cfg.pretrained_weights_fp / 'model.safetensors')
    model.load_state_dict(checkpoint)


    # # # calibrate on the tuning set
    # tuning_pyd = PytorchDataset(config=cfg.data_config, split="tuning")
    # tuning_loader = DataLoader(tuning_pyd, batch_size=8, collate_fn=tuning_pyd.collate, shuffle=False)
    # iso_reg = fit_isotonic_regression(tuning_loader, model, device="cpu")


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
    tuning_pyd = PytorchDataset(config=cfg.data_config, split="train")

    tuning_dataloader = DataLoader(
        tuning_pyd, batch_size=8, collate_fn=tuning_pyd.collate, shuffle=False
    )  

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
        # ap = []
        # auroc = []
        all_event_label_labels = []
        all_event_label_preds = []
        for batch in tuning_dataloader:
            encoded = model.encoder(batch,
                                        use_cache = None,
                                        output_attentions=False,
                                        output_hidden_states=True,
                                        return_dict=True)
            # accuracy.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.task_accuracy)
            # auroc.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.task_AUROC.item())
            # ap.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.average_precision.item())
        
            out_tuple = model.output_layer(batch, encoded.last_hidden_state, is_generation=True)
            event_label_labels = out_tuple[4]
            event_label_preds = out_tuple[3]
           
            all_event_label_labels.append(event_label_labels.detach().cpu())
            all_event_label_preds.append(event_label_preds.detach().cpu())
           
            accuracy.append(out_tuple[0].losses.task_accuracy)
            
        recall_fn = torchmetrics.classification.BinaryRecall(threshold=threshold_)
        precision_fn = torchmetrics.classification.BinaryPrecision(threshold=threshold_)
        f1_fn = torchmetrics.classification.BinaryF1Score(threshold=threshold_)
        ap_fn = torchmetrics.AveragePrecision(num_classes=2,task="binary")
    
       
        all_event_label_labels_ = torch.cat(all_event_label_labels[:-1], dim=1)
        all_event_label_preds_ = torch.cat(all_event_label_preds[:-1], dim=1)
        all_event_label_labels = all_event_label_labels_.view(-1, 2)
        all_event_label_preds = all_event_label_preds_.view(-1, 2)
        print(sum(all_event_label_preds[:, 1]))
        print(sum(all_event_label_preds[:, 0]))
        print(sum(all_event_label_labels[:, 1]))
        print(sum(all_event_label_labels[:, 0]))
        recall = recall_fn(all_event_label_preds, all_event_label_labels)
        ap = ap_fn(all_event_label_preds.to(dtype=torch.float),all_event_label_labels)
        precision = precision_fn(all_event_label_preds, all_event_label_labels)
        f1 = f1_fn(all_event_label_preds, all_event_label_labels)

        
        print('-------------------------------------------------------------------')
        print('METRICS:')
        print('-------------------------------------------------------------------')
        print(f'Mean F1 evaluated on TRAIN set: {f1}')
        print(f'Mean RECALL evaluated on TRAIN set: {recall}')
        print(f'Mean PRECISION evaluated on TRAIN set: {precision}')
        
        print(f'Mean AP evaluated on TRAIN set: {ap}')
        print('-------------------------------------------------------------------')
        print(f'{precision:.4f}')
        print(f'{recall:.4f}')

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
        all_interruption_labels = []
        all_interruption_preds = []
        for batch in tuning_dataloader:
            encoded = model.encoder(batch,
                                        use_cache = None,
                                        output_attentions=False,
                                        output_hidden_states=True,
                                        return_dict=True)
            
            out_tuple = model.output_layer(batch, encoded.last_hidden_state, is_generation=True)
            interruption_preds = out_tuple[5]
            interruption_labels = out_tuple[6]
            
            all_interruption_labels.append(interruption_labels.detach().cpu())
            all_interruption_preds.append(interruption_preds.detach().cpu())
           
            accuracy.append(out_tuple[0].losses.task_accuracy)
            
        recall_fn = torchmetrics.classification.BinaryRecall(threshold=threshold_)
        ap_fn = torchmetrics.AveragePrecision(num_classes=2,task="binary")
        precision_fn = torchmetrics.classification.BinaryPrecision(threshold=threshold_)
        f1_fn = torchmetrics.classification.BinaryF1Score(threshold=threshold_)
    
       
        all_interruption_labels = torch.cat(all_interruption_labels, dim=0)
        all_interruption_preds = torch.cat(all_interruption_preds, dim=0)
        # all_interruption_labels = all_interruption_labels_.view(-1, 2)
        # all_interruption_preds = all_interruption_preds_.view(-1, 2)

        print(sum(all_interruption_labels))
        # raw_probs = all_interruption_preds_.view(-1,2)[:, 1].numpy()
        # calibrated_probs = iso_reg.predict(raw_probs)
        # all_interruption_preds = torch.tensor(calibrated_probs).unsqueeze(1)
        # all_interruption_labels = all_interruption_labels_.view(-1, 2)[:, 1].unsqueeze(1)
        


        recall = recall_fn(all_interruption_preds, all_interruption_labels)
        ap = ap_fn(all_interruption_preds.to(dtype=torch.float),all_interruption_labels )
        precision = precision_fn(all_interruption_preds, all_interruption_labels)
        f1 = f1_fn(all_interruption_preds, all_interruption_labels)
        out_tuple = model.output_layer(batch, encoded.last_hidden_state, is_generation=True)
        print('-------------------------------------------------------------------')
        print('METRICS:')
        print('-------------------------------------------------------------------')
        print(f'Mean Accuracy evaluated on TRAIN set: {np.mean(accuracy):.4f}')
        print(f'Mean F1 evaluated on TRAIN set: {f1:.4f}')
        print(f'Mean RECALL evaluated on TRAIN set: {recall:.4f}')
        print(f'Mean AP evaluated on TRAIN set: {ap:.4f}')
        print(f'Mean PRECISION evaluated on TRAIN set: {precision:.4f}')
        print('-------------------------------------------------------------------')
        print(f'{precision:.4f}')
        print(f'{recall:.4f}')

    elif cfg.task_type == "class_dist":
        # Metrics:
        # 1. MSE
        # Return:
        # 1. class_dist_mse

        mse = []
        for batch in tuning_dataloader:
            encoded = model.encoder(batch,
                                        use_cache = None,
                                        output_attentions=False,
                                        output_hidden_states=True,
                                        return_dict=True)
            
            mse.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=False)[0].losses.mse)
        print('-------------------------------------------------------------------')
        print('METRICS:')
        print('-------------------------------------------------------------------')
        print(f'Mean MSE evaluated on TRAIN set: {np.mean(mse):.8f}')
        print('-------------------------------------------------------------------')
    elif cfg.task_type == "tti":
        # Metrics:
        # 1. MSE
        # Return:
        # 1. class_dist_mse
        mse = []
        for batch in tuning_dataloader:
            encoded = model.encoder(batch,
                                        use_cache = None,
                                        output_attentions=False,
                                        output_hidden_states=True,
                                        return_dict=True)

            mse.append(model.output_layer(batch, encoded.last_hidden_state, is_generation=True)[0].losses.mse)
        print('-------------------------------------------------------------------')
        print('METRICS:')
        print('-------------------------------------------------------------------')
        print(f'Mean MSE evaluated on TRAIN set: {np.mean(mse):.4f}')
        print('-------------------------------------------------------------------')
    

@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: FinetuneConfig):
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")


    return inference(cfg)



if __name__ == "__main__":
    main()