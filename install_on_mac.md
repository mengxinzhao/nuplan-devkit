on MacBook Pro M4 

pip install -r requirements_torch.txt doesn't work for me because the target is intel linux

instead, just run 
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

install other pytorch things
```
  pip install --upgrade pytorch-lightning
  pip install tensorboard
```
