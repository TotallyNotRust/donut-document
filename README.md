# Case
This ai should be able to read and understand invoices.\
It should give me information about what was purchased, how much each item cost, how much tax was paid, which vendor sent the invoice, and as much other information as possible.\

I will be using the Donut model to accomplish this task. I have chosen this model because it can scan documents much faster as it doesnt use OCR scanning like most other document scanning models. \

In this example ive used Pytorch Lightning for training, which is an easy to use deep learning training library

# Getting started

Run \
```
python3 -m pip install transformers datasets sentencepiece donut-python pytorch-lightning wandb datasets
```


Then simply run (There is no gui, so it will train a new file if no model is found in the directory) \
```python3 main.py [-a (accelerator)]```

-a or --accelerator can be passed to choose what accelerator to use, valid inputs are "cpu", "gpu" and "cuda".
By default "cpu" will be used.
It is reccomended to use either "gpu" or "cuda" if your system supports it.

# Based on this tutorial
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb

# Uses this dataset
https://huggingface.co/datasets/naver-clova-ix/cord-v2
