## Dataset Setup

To setup the dataset unzip the dataset `.zip` file into `../datasets/duckietown2` and run the following command:
```
python process_dataset.py
```

A new folder will be created called `duckietown2_processed`.

## Training

To train with the original dataset, run the following command:
```
python ducky/train.py --data duckietown.yaml --cfg yoloducky.yaml --img 256 --freeze 10 --epochs {CHOOSE} --weights yolov5s.pt
```

To train with our custom dataset, run the following command:
```
python ducky/train.py --data duckietown2.yaml --cfg yoloducky.yaml --img 256 --freeze 10 --epochs {CHOOSE} --hyp hyp.ducky.yaml --weights runs/train/exp{INSERT}/weights/best.pt
```

