# TDAF-for-CTR-prediction
An implementation for the paper: CTR Prediction with Temporal Drift of Clicking Features

Create a folder named 'train_output', navigate into it, and then create another folder named 'results' within it.


Configure script parameters to run:

--data_dir=../dataset --gpu 0 --algorithm DeepFM --dataset EDGRotatedMNIST --test_env 8 --steps 5001 --hparams {\"env_number\":9} --ctr_dataset douban --IL erm
