# usage:
#  --train_dir --val_dir

# this script was written to the Keras 1.x spec.  A number of important changes
# were introduced in Keras 2.0.  See release notes here:
# https://github.com/keras-team/keras/wiki/Keras-2.0-release-notes

import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172
STEPS_PER_EPOCH = 780

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.outputs
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.inputs, outputs=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  # nb_train_samples = get_nb_files(args.train_dir) #RC: don't believe this is needed
  nb_classes = len(glob.glob(args.train_dir + "/*"))
  validation_steps = get_nb_files(args.val_dir) #RC: use the entire validation data set to evaluate error/loss
  epochs = int(args.epochs)
  batch_size = int(args.batch_size)

  # data prep
  train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )
  test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

  # Takes the path to a directory, and generates batches of
  # augmented/normalized data
  train_generator = train_datagen.flow_from_directory(
    args.train_dir,
    # we specify the dimensions to which all images found will be resized
    # see https://keras.io/preprocessing/image/
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  validation_generator = test_datagen.flow_from_directory(
    args.val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  # setup model
  # use the flag include_top=False to leave out the weights of the last fully
  # connected layer since that is specific to the ImageNet competition, from
  # which the weights were previously trained
  base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)

  # transfer learning
  setup_to_transfer_learn(model, base_model)

  history_tl = model.fit_generator(
    train_generator,
    epochs=epochs,
    steps_per_epoch=STEPS_PER_EPOCH, # based on the original batch size of 32
                                # an epoch in training is 25,000. 25000/32 =  780
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight='auto')

  # fine-tuning
  setup_to_finetune(model)

  history_ft = model.fit_generator(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH, # based on the original batch size of 32
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight='auto')

  model.save(args.output_model_file)

  if args.plot:
    plot_training(history_ft)


def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--train_dir")
  a.add_argument("--val_dir")
  a.add_argument("--epochs", default=EPOCHS)
  a.add_argument("--batch_size", default=BAT_SIZE)
  a.add_argument("--output_model_file", default="inceptionv3-ft.model")
  a.add_argument("--plot", action="store_true")

  args = a.parse_args()
  if args.train_dir is None or args.val_dir is None:
    a.print_help()
    sys.exit(1)

  if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
    print("directories do not exist")
    sys.exit(1)

  train(args)
