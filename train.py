import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils import image_segmentation_generator
from LR_ASPP import LiteRASSP

N_CLASS = 2
INPUT_SIZE = 1024
epochs = 20
steps_per_epoch = 512
# latest_weights = tf.train.latest_checkpoint(f"checkpoints_{INPUT_SIZE}")
model = LiteRASSP(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), n_class=N_CLASS, alpha=1.0, weights=None).build()
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr=0.001),
              metrics=["accuracy"])
# model.load_weights(latest_weights)
train_gen = image_segmentation_generator(
    images_path="dataset/images", segs_path="dataset/annotations",  batch_size=2,  n_classes=N_CLASS,
    input_height=INPUT_SIZE, input_width=INPUT_SIZE, output_height=INPUT_SIZE, output_width=INPUT_SIZE)


for ep in range(epochs):
    print("Starting Epoch ", ep)
    model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=1)
    model.save_weights(f"checkpoints_{INPUT_SIZE}_new/cars.ckpt" + "." + str(ep))
    print("saved ", f"checkpoints_{INPUT_SIZE}_new/cars.ckpt" + ".model." + str(ep))
    print("Finished Epoch", ep)
