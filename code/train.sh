unzip -o -d ../tcdata/地表建筑物识别 ../tcdata/地表建筑物识别/train.zip
python ./train/data.py
python ./train/train_b8_1.py
python ./train/train_b8_2.py