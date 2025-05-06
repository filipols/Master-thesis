from omegaconf import OmegaConf
from torch.utils.data import DataLoader
# from EventStream.transformer.lightning_modules.fine_tuning import FinetuneConfig
from EventStream.transformer.lightning_modules.fine_tuning_dev import FinetuneConfig
from EventStream.data.pytorch_dataset import PytorchDataset
from EventStream.data.dataset_polars import Dataset
from EventStream.transformer.lightning_modules.generative_modeling import ESTForGenerativeSequenceModelingLM
from EventStream.transformer.generation.generation_utils import StructuredGenerationMixin
import polars as pl
from EventStream.transformer.conditionally_independent_model import CIPPTForGenerativeSequenceModeling 

# Initialize the config, overwriting the `max_seq_len` argument to a smaller value for the `data_config` to
# account for the elements you'll generate within the model's maximum sequence length.
cfg = FinetuneConfig(
    load_from_model_dir= '/home/filip-marcus/models/ESGPT_new/EventStreamGPT/pretrain/2025-02-19_08-54-50',
    task_df_name= 'task_df_ecom_cls_test',
)


# OmegaConf.set_struct(cfg, False)  # Allow changes to the config CHAT
cfg.data_config.save_dir = '/home/filip-marcus/models/ESGPT_new/EventStreamGPT/data/processed/test_ecom'
cfg.pretrained_weights_fp='/home/filip-marcus/models/ESGPT_new/EventStreamGPT/pretrain/2025-02-19_08-54-50/pretrained_weights'
cfg.data_config.do_include_start_time_min=True

#ESD = Dataset.load(cfg.data_config.save_dir)
train_pyd = PytorchDataset(cfg.data_config, split="train")

#print(ESD)
# events_df = ESD.task_df.lazy()
# print(events_df.collect())
# M = ESTForGenerativeSequenceModelingLM(
#     pretrained_weights_fp=cfg.pretrained_weights_fp, 
#     config=cfg.config, 
#     optimization_config=cfg.optimization_config, 
#     metrics_config=cfg.metrics_config
# )


M = CIPPTForGenerativeSequenceModeling(
    config=cfg.config
)



# M = ESTForGenerativeSequenceModeling.from_pretrained(
#     cfg.pretrained_weights_fp, config=cfg.config
# )


sample_dataloader = DataLoader(
    train_pyd, batch_size=2, collate_fn=train_pyd.collate, shuffle=False
)

sample_batch = next(iter(sample_dataloader))


generated = M.generate(
    sample_batch,
    max_new_events=2,  # Note that this must be within the model's `max_seq_len` - the input data length
    do_sample=True,
    return_dict_in_generate=True,
    output_scores=True,
)

print('ORIG SHAPE: ', sample_batch)
print('GENERATED:', generated.batch)

# generated.batch contains an extended PytorchBatch object with both the original data and
# the new, generated data

