# Traffic Map Forecasting

This repository contains the source code for CS680 Project

Build based on MIE lab's work (https://github.com/mie-lab/traffic4cast)

## What's new

- Modify data loader for traffic4cast 2020 challenge

<img src="https://github.com/paulwong16/TrafficMapForecasting/blob/master/datasets.png" alt="data" width="800"/>

- Add new models including Res UNet

- Add a fusion part for static and dynamic information

<img src="https://github.com/paulwong16/TrafficMapForecasting/blob/master/nn_title.png" alt="overview" width="800"/>

## How to

- Create 2 directory `data_new` and `data_new_raw`

- Copy the competition raw data into data_new_raw folder such that the top-level city folders are visible.

- Without static information

    - run `training_unet.py` and `training_deep_resunet.py`

- Fuse static information

    - run `training_unet_fusion.py` and `training_res_unet_fusion.py`
    
## Result (Berlin)

- Loss curve

<img src="https://github.com/paulwong16/TrafficMapForecasting/blob/master/loss.png" alt="loss" width="800"/>

- Error plot

- Loss curve

<img src="https://github.com/paulwong16/TrafficMapForecasting/blob/master/bar_plot_with_error_bars.png" alt="error" width="800"/>
