import numpy as np
import librosa
import os
import soundfile as sf
import pyloudnorm as pyln


def snr_mixer(clean, noise, snr):
    """
    Creates a mix of clean and noise with given SNR
    """
    amp_noise = np.linalg.norm(clean) / 10**(snr / 20)
    noise_norm = (noise / np.linalg.norm(noise)) * amp_noise
    mix = clean + noise_norm
    return mix


def vad_merge(w, top_db):
    """
    Deletes silence, which is audio below top_db, and merges loud chunks of audio
    """
    intervals = librosa.effects.split(w, top_db=top_db)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def cut_audios(s1, s2, sec, sr):
    """
    Cuts s1 and s2 in array of chunks of sec length
    """
    cut_len = sr * sec
    len1 = len(s1)
    len2 = len(s2)

    s1_cut = []
    s2_cut = []

    segment = 0
    while (segment + 1) * cut_len < len1 and (segment + 1) * cut_len < len2:
        s1_cut.append(s1[segment * cut_len:(segment + 1) * cut_len])
        s2_cut.append(s2[segment * cut_len:(segment + 1) * cut_len])

        segment += 1

    return s1_cut, s2_cut



def fix_length(s1, s2, min_or_max='max'):
    """
    Trims s1 and s2 to minimum length or pads to the maximum length of two audios
    """
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    return s1, s2


def create_mix(idx, triplet, snr_levels, out_dir, test=False, with_text=False, sr=16000, **kwargs):
    trim_db, vad_db = kwargs["trim_db"], kwargs["vad_db"]
    audioLen = kwargs["audioLen"]

    s1_path = triplet["target"]
    s2_path = triplet["noise"]
    ref_path = triplet["reference"]
    target_id = triplet["target_id"]
    noise_id = triplet["noise_id"]

    if with_text:
        text = triplet["text"]

    # Reading triplet files
    s1, _ = sf.read(os.path.join('', s1_path))
    s2, _ = sf.read(os.path.join('', s2_path))
    ref, _ = sf.read(os.path.join('', ref_path))

    # Normalize audio by loudness
    meter = pyln.Meter(sr) # create BS.1770 meter

    louds1 = meter.integrated_loudness(s1)
    louds2 = meter.integrated_loudness(s2)
    loudsRef = meter.integrated_loudness(ref)

    s1Norm = pyln.normalize.loudness(s1, louds1, -29)
    s2Norm = pyln.normalize.loudness(s2, louds2, -29)
    refNorm = pyln.normalize.loudness(ref, loudsRef, -23.0)

    amp_s1 = np.max(np.abs(s1Norm))
    amp_s2 = np.max(np.abs(s2Norm))
    amp_ref = np.max(np.abs(refNorm))

    if amp_s1 == 0 or amp_s2 == 0 or amp_ref == 0:
        return

    # Trim audio at the beginning and at the end
    if trim_db:
        ref, _ = librosa.effects.trim(refNorm, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1Norm, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2Norm, top_db=trim_db)

    if len(ref) < sr:
        return

    # Paths to output files
    path_mix = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-mixed.wav")
    path_target = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-target.wav")
    path_ref = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-ref.wav")
    if with_text:
        path_text = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + ".txt")

    # Choose desired snr level for noise
    # Use as an augmentation
    snr = np.random.choice(snr_levels, 1).item()

    if not test:
        # Remove silence
        s1, s2 = vad_merge(s1, vad_db), vad_merge(s2, vad_db)
        # Cut by audioLen chunks
        s1_cut, s2_cut = cut_audios(s1, s2, audioLen, sr)

        for i in range(len(s1_cut)):
            # Mix two chunks
            mix = snr_mixer(s1_cut[i], s2_cut[i], snr)
            # Normalize target audio 
            louds1 = meter.integrated_loudness(s1_cut[i])
            s1_cut[i] = pyln.normalize.loudness(s1_cut[i], louds1, -23.0)
            # Normalize mix audio
            loudMix = meter.integrated_loudness(mix)
            mix = pyln.normalize.loudness(mix, loudMix, -23.0)
            # Write output chunks
            path_mix_i = path_mix.replace("-mixed.wav", f"_{i}-mixed.wav")
            path_target_i = path_target.replace("-target.wav", f"_{i}-target.wav")
            path_ref_i = path_ref.replace("-ref.wav", f"_{i}-ref.wav")
            sf.write(path_mix_i, mix, sr)
            sf.write(path_target_i, s1_cut[i], sr)
            sf.write(path_ref_i, ref, sr)
    else:
        # Expand two audios by maximum length
        s1, s2 = fix_length(s1, s2, 'max')
        # Mix two audios
        mix = snr_mixer(s1, s2, snr)
        # Normalize target audio
        louds1 = meter.integrated_loudness(s1)
        s1 = pyln.normalize.loudness(s1, louds1, -23.0)
        # Normalize mix
        loudMix = meter.integrated_loudness(mix)
        mix = pyln.normalize.loudness(mix, loudMix, -23.0)
        # Write outputs
        sf.write(path_mix, mix, sr)
        sf.write(path_target, s1, sr)
        sf.write(path_ref, ref, sr)
        if with_text:
            with open(path_text, "w+") as f:
                f.write(text)