# Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
#training and evaluation (best configurations have been set as default)
python train_intent.py 
#testing
python3 test_intent.py --test_file {testing dataset path} --ckpt_path {best checkpoint path} --pred_file {predict file path}
```

## Slot tagging
```shell
#training and evaluation (best configurations have been set as default)
python train_slot.py 
#testing
python3 test_slot.py --test_file {testing dataset path} --ckpt_path {best checkpoint path} --pred_file {predict file path}
```

