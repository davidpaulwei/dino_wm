HYDRA_FULL_ERROR=1 python train.py --config-name train.yaml env=reach frameskip=5 num_hist=1

# to train decoder only, loading from a trained encoder-decoder pair:

HYDRA_FULL_ERROR=1 python train.py --config-name train.yaml env=pick_cup_decoder frameskip=5 num_hist=1 saved_folder="/home/thomas/dino_wm/outputs/2025-03-10/01-42-46" training.epochs=130 model.train_predictor=False