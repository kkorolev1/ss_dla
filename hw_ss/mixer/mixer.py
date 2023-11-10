import os
from glob import glob
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from hw_ss.mixer.utils import create_mix


class LibriSpeechSpeakerFiles:
    """
    Collects audio filepaths of a specific Librispeech speaker
    """
    def __init__(self, speaker_id, audios_dir, audioTemplate="*-norm.wav", with_text=False):
        self.id = speaker_id
        self.files = []
        self.audioTemplate = audioTemplate
        self.files = self.find_files_by_worker(audios_dir)
        self.texts = None

        if with_text:
            self.texts = self.find_text(audios_dir) # map from audiopath to text

    def find_files_by_worker(self, audios_dir):
        speakerDir = os.path.join(audios_dir, self.id)
        chapterDirs = os.scandir(speakerDir)
        files = []
        for chapterDir in chapterDirs:
            files = files + [file for file in glob(os.path.join(speakerDir, chapterDir.name, self.audioTemplate))]
        return files


    def find_text(self, audios_dir):
        speakerDir = os.path.join(audios_dir, self.id)
        chapterDirs = os.scandir(speakerDir)
        texts = {}
        for chapterDir in chapterDirs:
            trans_files = glob(os.path.join(speakerDir, chapterDir.name, "*.txt"))
            if len(trans_files) > 0:
                with open(trans_files[0]) as f:
                    for line in f.readlines():
                        splitted_line = line.split(" ")
                        audiopath = os.path.join(speakerDir, chapterDir.name, splitted_line[0])
                        text = " ".join(splitted_line[1:])
                        texts[audiopath] = text
        return texts
    

def collect_speakers_files(audios_dir, audioTemplate="*-norm.wav", with_text=False):
    result = []
    for speaker_id in os.listdir(audios_dir):
        speaker_dir_path = os.path.join(audios_dir, speaker_id)
        if os.path.isdir(speaker_dir_path):
            result.append(LibriSpeechSpeakerFiles(speaker_id, audios_dir, audioTemplate, with_text))
    return result


class MixtureGenerator:
    def __init__(self, speakers_files, out_folder, nfiles=5000, test=False, with_text=False, randomState=42):
        self.speakers_files = speakers_files # list of SpeakerFiles for every speaker_id
        self.nfiles = nfiles
        self.randomState = randomState
        self.out_folder = out_folder
        self.test = test
        random.seed(self.randomState)
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder, exist_ok=True)
        self.with_text = with_text


    def generate_triplets(self):
        """
        Generates nfiles triplets sampling two speakers
        and randomly choosing audios for them
        """
        i = 0
        all_triplets = {"reference": [], "target": [], "noise": [], "target_id": [], "noise_id": []}
        if self.with_text:
            all_triplets["text"] = []
        while i < self.nfiles:
            spk1, spk2 = random.sample(self.speakers_files, 2)

            if len(spk1.files) < 2 or len(spk2.files) < 2:
                continue

            target, reference = random.sample(spk1.files, 2)
            noise = random.choice(spk2.files)
            all_triplets["reference"].append(reference)
            all_triplets["target"].append(target)
            all_triplets["noise"].append(noise)
            all_triplets["target_id"].append(spk1.id)
            all_triplets["noise_id"].append(spk2.id)
            if self.with_text:
                target_path = Path(target)
                all_triplets["text"].append(spk1.texts[os.path.join(target_path.parent, target_path.stem)])
            i += 1

        return all_triplets


    def generate_mixes(self, snr_levels=[0], num_workers=10, update_steps=10, **kwargs):
        print(kwargs)
        print(f"snr_levels: {snr_levels}")
        triplets = self.generate_triplets()

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = []

            for i in range(self.nfiles):
                triplet = {"reference": triplets["reference"][i],
                           "target": triplets["target"][i],
                           "noise": triplets["noise"][i],
                           "target_id": triplets["target_id"][i],
                           "noise_id": triplets["noise_id"][i]}
                if self.with_text:
                    triplet["text"] = triplets["text"][i]
                futures.append(pool.submit(create_mix, i, triplet,
                                           snr_levels, self.out_folder,
                                           test=self.test, with_text=self.with_text, **kwargs))

            for i, future in enumerate(futures):
                future.result()
                if (i + 1) % max(self.nfiles // update_steps, 1) == 0:
                    print(f"Files Processed | {i + 1} out of {self.nfiles}")


speakers_files = collect_speakers_files("/home/korolevki/ss_dla/data/datasets/librispeech/test-clean", "*.flac", with_text=False)
out_folder = "/home/korolevki/ss_dla/data/datasets/librispeech/dummy-test"

mixer = MixtureGenerator(speakers_files, out_folder, nfiles=10, test=True, with_text=False)

mixer.generate_mixes(
    snr_levels=[0],
    num_workers=2,
    update_steps=100,
    trim_db=None,
    vad_db=20,
    audioLen=3
)