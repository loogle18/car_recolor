import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils import image_segmentation_generator
from LR_ASPP import LiteRASSP

epochs = 20
steps_per_epoch = 512
latest_weights = tf.train.latest_checkpoint("checkpoints")
model = LiteRASSP(input_shape=(1024, 1024, 3), n_class=2, alpha=1.0, weights=None).build()
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr=0.001),
              metrics=["accuracy"])
model.load_weights(latest_weights)
train_gen = image_segmentation_generator(
    images_path="cars/images", segs_path="cars/annotations",  batch_size=2,  n_classes=2,
    input_height=1024, input_width=1024, output_height=1024, output_width=1024)


for ep in range(epochs):
    print("Starting Epoch ", ep)
    model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=1)
    model.save_weights("checkpoints/cars.ckpt" + "." + str(ep))
    print("saved ", "checkpoints/cars.ckpt" + ".model." + str(ep))
    print("Finished Epoch", ep)
