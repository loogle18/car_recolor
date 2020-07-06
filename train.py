import glob
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils import image_segmentation_generator
from LR_ASPP import LiteRASSP

N_CHANNELS = 1
N_CLASS = 2
INPUT_SIZE = 1024
EPOCHS = 20
BATCH_SIZE = 2
TRAIN_SIZE = len(glob.glob("dataset/final/train/images/*"))
VALID_SIZE = len(glob.glob("dataset/final/valid/images/*"))
latest_weights = tf.train.latest_checkpoint(f"checkpoints/{INPUT_SIZE}_gs_pd_1298_158")
model = LiteRASSP(input_shape=(INPUT_SIZE, INPUT_SIZE, N_CHANNELS), n_class=N_CLASS, alpha=1.0, weights=None).build()
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr=0.00146),
              metrics=["accuracy"])
model.load_weights(latest_weights)
train_gen = image_segmentation_generator(images_path="dataset/final/train/images",
                                         segs_path="dataset/final/train/masks",
                                         batch_size=BATCH_SIZE,
                                         n_classes=N_CLASS,
                                         input_height=INPUT_SIZE,
                                         input_width=INPUT_SIZE,
                                         output_height=INPUT_SIZE,
                                         output_width=INPUT_SIZE)


valid_gen = image_segmentation_generator(images_path="dataset/final/valid/images",
                                         segs_path="dataset/final/valid/masks",
                                         batch_size=BATCH_SIZE,
                                         n_classes=N_CLASS,
                                         input_height=INPUT_SIZE,
                                         input_width=INPUT_SIZE,
                                         output_height=INPUT_SIZE,
                                         output_width=INPUT_SIZE)


for ep in range(EPOCHS):
    # ep = ep + 20
    print("Starting Epoch ", ep)
    model.fit(train_gen,
              steps_per_epoch=max(1, TRAIN_SIZE // BATCH_SIZE),
              # steps_per_epoch=512,
              validation_data=valid_gen,
              validation_steps=max(1, VALID_SIZE // BATCH_SIZE),
              epochs=1)
    model.save_weights(f"checkpoints/{INPUT_SIZE}_gs_pd_1298_158/cars.ckpt" + "." + str(ep))
    print("saved ", f"checkpoints/{INPUT_SIZE}_gs_pd_1298_158/cars.ckpt" + ".model." + str(ep))
    print("Finished Epoch", ep)
