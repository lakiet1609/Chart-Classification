# Chart Classification Pipeline

![newpipeline2](https://github.com/lakiet1609/Chart-Classification/assets/116550803/33218f46-ef8c-459a-b61e-4ff1b7de40b7)

Chart recognition in document images is important in digitizing and understanding document information. This process often includes two main processes: chart classification and chart understanding. Chart classification is critical to supporting input information processing. This paper proposes a novel method to classify the chart image by redefining class categories by composing multiple familiar and biased classes into larger, more representative clusters. Our approach involves a multi-step process wherein we amalgamate related classes to form composite classes, subsequently performing classification within these composed clusters. By iteratively refining classifications within each cluster, we effectively mitigate the effects of class imbalance and enhance the accuracy of predictions. Significantly, our method facilitates the improvement of classification accuracy for classes with similar features without necessitating additional data collection efforts. Our approach has been evaluated through empirical testing via a public dataset: UBPMC. The experimental results show the efficacy of our approach in handling imbalanced datasets and improving model performance in scenarios where traditional techniques may fall short.

## Install required software

Please run the following docker install commands:

```
docker build .
docker run -it --name chart --gpus all --shm-size=1gb -v YOUR_PATH vision:latest bash
```

## Download dataset + pre-trained models
Please download the pretrained model by following this link [located here](https://drive.google.com/drive/folders/1MN17L4FJ2DVZUcGJxl1b3CT5Xfdiyfyn?usp=sharing) and the training [located here](https://www.dropbox.com/s/85yfkigo5916xk1/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0.zip?dl=0) and testing [located here](https://www.dropbox.com/s/w0j0rxund06y04f/ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1.zip?dl=0) datasets 

Thereafter, unzip the training and testing datasets and place the contents of ```/training_set``` and ```/testing_set``` folder within the right format like below:

```bash
dataset/
    ├── train/
    │   ├── area/
    │   ├── surface/
    │   ├── heatmap/
    │   ├── line/
    │   ├── scatter line/
    │   └── .../
    │
    └── test/
        ├── area/
        ├── surface/
        ├── heatmap/
        ├── line/
        ├── scatter line/
        └── .../
```

```split_dataset.py``` can be used for organizing the origin dataset into the required format

## How to run
Please make sure that dataset and models are in the right format and directory befor execute any train or inference process

### Train the network
For training the each process, the model name is set as A, B, base ... Depends on each phase, we choose the right model for that phase
```
python train.py --model-name YOUR_PROCESS
```

### Run inference only (without training a new network)
Please make sure that the ```/dataset``` and ```/pretrained_models``` folders are correctly placed in the specific directory. If ```entity``` and ```/pretrained_models``` exists then we can modify it in ```load_entity``` and ```load_checkpoint```, else just run the pipeline from scratch. Thereafter, run the following command:

```
python predict.py --model-name YOUR_PROCESS 
```