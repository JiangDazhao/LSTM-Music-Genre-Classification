## Music Genre Classification with LSTMs
 * build a small speech recognition system using the LSTM model based on the Tensor-flow programing framework
 * Classify music files based on genre from the GTZAN music corpus
 * Use multiple layers of LSTM Recurrent Neural Nets

### Dataset
 recognize 5 different NAMEs, Record each NAME for at least 22 times (10 for training, 5 for test, 5 for validation, and 2 for prediction test), so you need at least 22*5 files in total


### Train the LSTM model
delete `stm_genre_classifier_lstm.h5` and all `.npy` files under `\gtzan`， they may come from previous training tests

    python lstm_genre_classifier_keras.py

You will obtain the accuracy and loss parameters

The model `model.json` and weight file (`lstm_genre_classifier_lstm.h5`) are created, 
 * copy `model.json` to `\weights\model.json` (overwrite if necessary).
 * rename `lstm_genre_classifier_lstm.h5` to become `model_weights.h5`, and copy `model_weights.h5` to `\weights\ model_weights.h5` (overwrite if necessary)

### Test trained LSTM model


 In the `./weights/` you can find trained model weights and model architecture.

 To test the model on your custom audio file, run

     python3 predict_example.py path/to/custom/file.mp3
 or to test the model on our custom files, run

     python3 predict_example.py audio/classical_music.mp3

### Ideas for improving accuracy:
 * Normalize MFCCs & other input features ([Recurrent BatchNorm](https://arxiv.org/pdf/1603.09025v4.pdf)?)
 * Decay learning rate
 * How are we initing the weights?
 * Better optimization hyperparameters (too little dropout)
 * Do you have avoidable bias? How's your variance?