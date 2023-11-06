import logging
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    # TODO: your code here
    
    result_batch["reference"] = pad_sequence([item["reference"].squeeze(0)
                                                for item in dataset_items], batch_first=True
                                              ).unsqueeze(1)
    result_batch["reference_length"] = torch.tensor([item["reference"].shape[-1] for item in dataset_items])
    result_batch["mix"] = pad_sequence([item["mix"].squeeze(0)
                                            for item in dataset_items], batch_first=True
                                          ).unsqueeze(1)
    result_batch["target"] = pad_sequence([item["target"].squeeze(0)
                                            for item in dataset_items], batch_first=True
                                          ).unsqueeze(1)
    result_batch["speaker_id"] = torch.tensor([item["speaker_id"] for item in dataset_items])
    result_batch["mix_path"] = [item["mix_path"] for item in dataset_items]
    return result_batch