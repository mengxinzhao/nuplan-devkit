on MacBook Pro M4 
1. have to use python3.9 because M4 default python is 3.13.5 which makes some of nuplan requirement packages not working
2. pip install -r requirements_torch.txt doesn't work for me because the target is intel linux

instead, just run 
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

install other pytorch things
```
  pip install --upgrade pytorch-lightning
  pip install tensorboard
```
3.  install python packages separately for below two.
pip install guppy3
pip install hydra-core
then pip install -r requirements.txt
numpy version is too slow because numpy>2 doesn't have bool8 anymore
