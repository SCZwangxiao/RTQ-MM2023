
Go to the [dataset page](https://github.com/doc-doc/NExT-QA) to download [raw videos and annotations](https://drive.google.com/drive/folders/1gKRR2es8-gRTyP25CvrrVtV6aN5UxttF) in the following:
```bash
NExTVideo.zip
nextqa.zip
test-data-nextqa.zip
```


Unzip them:
```bash
unzip NExTVideo.zip
unzip nextqa.zip
unzip test-data-nextqa.zip
```


Process them:
```bash
python convert_data_format.py [PATH TO DATASET]
```