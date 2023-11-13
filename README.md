# SS HW 2

Source separation task using SpEx+ model trained on mixes based on Librispeech speakers. The task was to separate two speakers with a reference audio of a target speaker.

## Installation guide

```shell
pip install -r ./requirements.txt
```

Best model can be found [here](https://drive.google.com/file/d/136cmgBobfAbA5mFrnvq9pKwlXhtg2RlO/view?usp=sharing). All parameters of the model are set to the default values in a SpexPlus class. For the test you might need to set only num_speakers=251. 

Configs can be found in hw_ss/configs folder. In particular, for testing use config_test.json.

## Training
Trained on [this dataset](https://www.kaggle.com/datasets/lizakonstantinova/librispeech-mixes).
```shell
python train.py -c CONFIG
```

## Testing
Validated on [this dataset](https://www.kaggle.com/datasets/kafafyf/librispeech-mixes-test).
```shell
python test.py -c CONFIG -r CHECKPOINT -t TEST_DIRECTORY
```

||SI-SDR|PESQ|
|---|---|---|
|Val set|11.23|2.14|
