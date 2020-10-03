import tensorflow as tf

training_accuracy_needed = 0.96


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > training_accuracy_needed:
            print("\n Reached " + str(training_accuracy_needed * 100) + "% accuracy so cancelling Training")
            self.model.stop_training = True
