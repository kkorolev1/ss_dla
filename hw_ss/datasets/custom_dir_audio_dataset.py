import logging
from pathlib import Path
from glob import glob
import os
from hw_ss.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


def id_to_path(files):
    """
    Returns a mapping from ID to audio path
    """
    return {os.path.basename(path).split("-")[0]: path for path in files}


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, mix_dir, ref_dir, target_dir, *args, **kwargs):
        data = []
        mixes = id_to_path(glob(os.path.join(mix_dir, '*-mixed.wav')))
        refs = id_to_path(glob(os.path.join(ref_dir, '*-ref.wav')))
        targets = id_to_path(glob(os.path.join(target_dir, '*-target.wav')))

        for id in (mixes.keys() & refs.keys() & targets.keys()):
            data.append({
                "mix": mixes[id],
                "reference": refs[id],
                "target": targets[id],
                "speaker_id": 0 # dummy speaker_id, which is not used during testing
            })
        super().__init__(data, *args, **kwargs)