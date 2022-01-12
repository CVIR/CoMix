## Background Extraction using Temporal Median Filtering.

### Running instructions

Example code for the UCF dataset:

```
python extract_bg_tmf.py -i ./../video_splits/ucf101_train_hmdb_ucf.csv --input_dir ./../data/ucf_hmdb/ucf_videos --output_dir ./../data/ucf_hmdb/ucf_BG -r 224 -offset 1
```

Example code for the Epic-Kitchens (D1) dataset:

```
python extract_bg_tmf.py -i ./../video_splits/D1_train.pkl --input_dir ./../data/epic_kitchens/epic_kitchens_videos --output_dir ./../data/epic_kitchens/epic_kitchens_D1_BG -r 224 -offset 1
```

Follow the same Input directory structure as mentioned in the main code (`./../README.md`).

### Additional installations
1) Install `ffmpeg`
2) `pip install -r bg_requirements.txt`

### Acknowledgements

Code uses portions of https://github.com/Pantsworth/temporal-median-video
