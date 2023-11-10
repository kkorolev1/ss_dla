# SS HW 2

Source separation task using SpEx+ model trained on mixes based on Librispeech speakers. The task was to separate two speakers with a reference audio of a target speaker.

## Installation guide

```shell
pip install -r ./requirements.txt
```

## Training
```shell
python train.py -c CONFIG
```

## Testing
```shell
python test.py -c CONFIG -r CHECKPOINT -t TEST_DIRECTORY
```
