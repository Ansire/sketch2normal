# sketch2normal

This is the implementation of the paper [Interactive Sketch-Based Normal Map Generation with Deep Neural Networks](http://sweb.cityu.edu.hk/hongbofu/doc/sketch2normal_i3D2018.pdf)

<img src="./teaser.PNG" width="700px"/>

## Setup

We run the program on a Linux desktop using python.

## Usage

- Train the model:
```bash
pyhton main.py --phase train --dataset_name <dataset>
```

- Test:
```bash
python main.py --phase test --dataset_name <dataset>
```

- Visualize the training process:
```bash
cd logs
tensorboard --logdir=./
```

- Data: TBA

## Acknowledgement
This code is based on the implementation of pix2pix from [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow). Thanks for the great work!