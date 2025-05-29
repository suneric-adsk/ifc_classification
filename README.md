# ifc_classification
Classification task on IFC entities

## Exploration based on [IFCNet](https://ifcnet.e3d.rwth-aachen.de/)

## Preparation

### Environment setup
On Ubuntu 22.04

```
git clone https://github.com/suneric-adsk/ifc_classification.git
cd ifc_classification
conda env create -f environment.yml
conda activate ifcnet
```

### Dataset
Download dataset from original paper, and put them into "dataset" folder


## Training
For MVCNN
```
python classification.py train --type view --epochs 30

```
For DGCNN
```
python classification.py train --type point --epochs 300
```

For MeshNet
```
python classification.py train --type mesh --epochs 300
```

For UVNet
```
python classification.py train --type brep --epochs 300
```

The logs and checkpoints will be stored in "results" folder and can be monitored with Tensorboard
```
tensorboard --logdir results
```

## Test
The best checkpoints can be used to test the models
```
python classification.py test --type <view|ply|mesh> --ckpt <path to best.ckpt>
``` 

