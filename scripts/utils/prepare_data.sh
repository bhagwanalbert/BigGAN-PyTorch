#!/bin/bash
python make_hdf5.py --dataset CORE50 --batch_size 100 --data_root data
python calculate_inception_moments.py --dataset CORE50_hdf5 --data_root data
