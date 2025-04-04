# Model-Incremental Few-shot Learning with Distribution Alignment Prompts for Infrared Small Target Detection

## Model
|![MIFL](./Fig/model.png)|
|:--:|
|*MIFL*|

## results on IRDST, DAUB and ITSDT datasets

| ![Vis](./Fig/vis.png) |
|:--:|
| *visiualization* |

## Datasets
### 1. [IRDST](https://xzbai.buaa.edu.cn/datasets.html), [DUAB](https://www.scidb.cn/en/detail?dataSetId=720626420933459968) and [ITSDT](https://www.scidb.cn/en/detail?dataSetId=de971a1898774dc5921b68793817916e&dataSetType=journal)


### 2. The COCO format need to convert to txt format.
``` python 
python utils_coco/coco_to_txt.py
```
### 3. The class of dataset should write to a txt file. 
Such as model_data/classes.txt

## Train
## The base task.
``` python 
python train_base.py
```
## The second task.
``` python 
python train_2.py
```
## The more tasks.
``` python 
python train_res.py
```
## Evaluate
##Evaluate single task
```python 
python vid_map_coco.py
```

##Evaluate all tasks
```python 
python test_incre.py
```
##Evaluate descriptors
```python 
python test_descriptor.py
```


## Reference
https://github.com/bubbliiiing/yolox-pytorch/
