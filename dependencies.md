# Remote Dependencies

- speech-dataset-parser
- text-utils
  - pronunciation_dict_parser
  - g2p_en
  - sentence2pronunciation
- audio-utils
- image-utils

## Pipfile

### Local

```Pipfile
speech-dataset-parser = {editable = true, path = "./../speech-dataset-parser"}
text-utils = {editable = true, path = "./../text-utils"}
audio-utils = {editable = true, path = "./../audio-utils"}
image-utils = {editable = true, path = "./../image-utils"}

pronunciation_dict_parser = {editable = true, path = "./../pronunciation_dict_parser"}
g2p_en = {editable = true, path = "./../g2p"}
sentence2pronunciation = {editable = true, path = "./../sentence2pronunciation"}
```

### Remote

```Pipfile
speech-dataset-parser = {editable = true, ref = "master", git = "https://github.com/stefantaubert/speech-dataset-parser.git"}
text-utils = {editable = true, ref = "master", git = "https://github.com/stefantaubert/text-utils.git"}
audio-utils = {editable = true, ref = "master", git = "https://github.com/stefantaubert/audio-utils.git"}
image-utils = {editable = true, ref = "master", git = "https://github.com/stefantaubert/image-utils.git"}
```

## setup.cfg

```cfg
speech_dataset_parser@git+https://github.com/stefantaubert/speech-dataset-parser.git@master
text_utils@git+https://github.com/stefantaubert/text-utils.git@master
audio_utils@git+https://github.com/stefantaubert/audio-utils.git@master
image_utils@git+https://github.com/stefantaubert/image-utils.git@master
```
