# LULC abundance estimation of multi-spectral time series + ancillary data of MODIS pixels from Andalusia with AI.	


## Background
Land Use and Land Cover (LULC) maps are of paramount importance to provide precise information for dynamic monitoring, planning, and management of the Earth. Regularly updated global LULC datasets provide the basis for understanding the status, trends, and pressures of human activity on carbon cycles, biodiversity, and other natural and anthropogenic processes. Because of that, being able to automatically create these maps without human labor by using new Remote Sensing (RS) and Artificial Intelligence (AI) technologies is a great avenue to explore.


## Introduction
In the last few decades, LULC maps have been created using RS images following the "raster data model", where the Earth's surface is divided in squares of a certain spatial resolution called pixels. Then, each of these pixels is assigned a "LULC class" (e.g, forest, water, urban...) that represents the underlying type of the Earth surface in each pixel. The number of different classes of a LULC map is referred to as thematic resolution. Frequently, the spatial and thematic resolutions do not match, which leads to the mixed pixel problem, i.e., pixels are not pure but contain several LULC classes. Under a "hard" classification approach, a mixed pixel would be assigned just one LULC class (e.g., the dominant class) while under a "soft" classification approach (also called spectral unmixing or abundance estimation) the relative abundance of each LULC class is provided per pixel. 

Moreover, ancillary information regarding the geographic, topographic and climatic information of the studied area could also be useful to classify each pixel to its corresponding LULC class. Concretely, the following ancillary variables are studied: GPS coordinates, altitude, slope, precipitation, potential evapotranspiration, mean temperature, maximum temperature and minimum temperature.

## Aims

To estimate the relative abundance of LULC classes in Andalusia and develop an AI model to automatically perform the task, a new labeled dataset of Andalusia of pixels from MODIS at 460m resolution was built. Each pixel is a multi-spectral time series and includes the corresponding ancillary information. Also, each pixel is labeled with its corresponding LULC class abundances inside that pixel. The label is provided at two hierarchical levels, namely N1 (coarser) and N2 (finer). To create these labels, the SIPNA (Sistema de Información sobre el Patrimonio Natural de Andalucía) product was used, which aims to build an information system on the natural heritage of Andalusia. The first level "N1" contains four high level LULC classes, whereas the second level "N2" contains ten finer LULC classes. Thus, this model was mainly trained and validated in the region of Andalusia in Spain. Once the dataset was created, the AI model was trained using about 80% of the data, and then validated with the remaining 20% following a carefully spatial block splitting strategy to avoid spatial autocorrelation. The AI model processes the multi-spectral time series from MODIS at 460m and the ancillary information to predict the LULC abundances in that pixel. 

Both, the RS dataset with the ancillary data used to create the AI model and the AI model itself are the deliverables of this project. In summary, we provide an automatic tool to estimate the LULC classes abundances of MODIS pixels from Andalusia using a soft classification approach and set a methodology that could be applied to other satellites where a better spatial resolution allows the use of more fine LULC classes in the future. Also, the AI model could serve as a starting point for researchers interested in applying the model in other locations, i.e, they can fine-tune the existing model with data for the new region of interest requiring far less training data thanks to transferring the learned patterns of our model.


## Setup
### Install requirements 
```bash
conda create -n msmtu python=3.8 -y
conda activate msmtu

# install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 -c pytorch

# install dependency and lavin
pip install -r requirements.txt
```

## Sample Execution

```bash
python predict.py --level N1 --output_dir ./outputs
```