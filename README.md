# Case
This ai should be able to read and understand invoices.\
It should give me information about what was purchased, how much each item cost, how much tax was paid, which vendor sent the invoice, and as much other information as possible.

I will be using the Donut model to accomplish this task. I have chosen this model because it can scan documents much faster as it doesnt use OCR scanning like most other document scanning models. 

The Donut model is a supervised deep learning model, this is of course because it uses a neural net to understand and read images, and is trained on labeled data.\
We use PyTorch Lightning in this project to train the model.

I have looked at many other solutions to the same issue over the last few years, and all of them have chosen to use ai to accomplish this task. I doubt it would ever be feasible to do without. 

You can find the model I trained at https://huggingface.co/TotallyNotRust/donut

# Getting started

Run \
```
python3 -m pip install transformers datasets sentencepiece donut-python pytorch-lightning wandb datasets
```

Then simply run to train a new model. \
```python3 main.py [-a (accelerator)]```

-a or --accelerator can be passed to choose what accelerator to use, valid inputs are "cpu", "gpu" and "cuda".
By default "cpu" will be used.
It is reccomended to use either "gpu" or "cuda" if your system supports it.

Then you can scan \
````python3 scan.py -f [file name]``

# Based on this tutorial
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb

# Uses this dataset
https://huggingface.co/datasets/naver-clova-ix/cord-v2
