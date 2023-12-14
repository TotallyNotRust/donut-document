

from datasets import load_dataset
from ast import literal_eval
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, EarlyStopping
import pytorch_lightning as pl

from model import DonutDataset
from lightningmodule import DonutModelPLModule

def main():
    dataset = load_dataset("naver-clova-ix/cord-v2")

    example = dataset['train'][0]
    image = example['image']
    # let's make the image a bit smaller when visualizing
    width, height = image.size
    ground_truth = example['ground_truth']

    image_size = [1280, 960]
    max_length = 768

    # update image_size of the encoder
    # during pre-training, a larger image size was used
    config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
    config.encoder.image_size = image_size # (height, width)
    # update max_length of the decoder (for generation)
    config.decoder.max_length = max_length

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)
        
    # we update some settings which differ from pretraining; namely the size of the images + no rotation required
    # source: https://github.com/clovaai/donut/blob/master/config/train_cord.yaml
    processor.image_processor.size = image_size[::-1] # should be (width, height)
    processor.image_processor.do_align_long_axis = False

    train_dataset = DonutDataset(config, "naver-clova-ix/cord-v2", max_length=max_length,
                                split="train", task_start_token="", prompt_end_token="",
                                sort_json_key=False, # cord dataset is preprocessed, so no need for this
                                )

    val_dataset = DonutDataset(config, "naver-clova-ix/cord-v2", max_length=max_length,
                                split="validation", task_start_token="", prompt_end_token="",
                                sort_json_key=False, # cord dataset is preprocessed, so no need for this
                                )

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([''])[0]



    # sanity check
    print("Pad token ID:", processor.decode([model.config.pad_token_id]))
    print("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))
        
    
    # feel free to increase the batch size if you have a lot of memory
    # I'm fine-tuning on Colab and given the large image size, batch size > 1 is not feasible
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print("DATALOADER INIT")
    batch = next(iter(train_dataloader))
    pixel_values, labels, target_sequences = batch
    print(pixel_values.shape)
    for id in labels.squeeze().tolist()[:30]:
        if id != -100:
            print(processor.decode([id]))
        else:
            print(id)

    print("DATALOADER DONE")
    # let's check the first validation batch
    batch = next(iter(val_dataloader))
    pixel_values, labels, target_sequences = batch
    print(pixel_values.shape)
    print("VALIDATOR DONE")

    print("Start training")
    train_config = {"max_epochs":30,
          "val_check_interval":0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch":1,
          "gradient_clip_val":1.0,
          "num_training_samples_per_epoch": 800,
          "lr":3e-5,
          "train_batch_sizes": [8],
          "val_batch_sizes": [1],
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": 300, # 800/8*30/10, 10%
          "result_path": "./result",
          "verbose": True,
          }
    model_module = DonutModelPLModule(train_config, processor, model, train_dataloader, val_dataloader, max_length)

    early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

    trainer = pl.Trainer(
            accelerator="cpu",
            # accelerator = "gpu", # uncomment to use gpu
            devices=1,
            max_epochs=train_config.get("max_epochs"),
            val_check_interval=train_config.get("val_check_interval"),
            check_val_every_n_epoch=train_config.get("check_val_every_n_epoch"),
            gradient_clip_val=train_config.get("gradient_clip_val"),
            precision=16, # we'll use mixed precision
            num_sanity_val_steps=0,
            #logger=wandb_logger,
            callbacks=[early_stop_callback],
    )

    trainer.fit(model_module)

    print("Training complete")
        


if __name__ == "__main__":
    main()