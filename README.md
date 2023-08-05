# iSMART
**Intelligent System for bi-Modal Recognition of Personality Traits on First Impressions V2 dataset**

In this project, you are tasked with building a multimodal deep neural network for personality trait detection using tf.keras. You'll be working with the First Impressions V2 dataset, which contains short video clips of speakers talking to the camera. The goal is to predict five personality traits: Extraversion, Agreeableness, Conscientiousness, Neuroticism, and Openness.

To prepare the dataset, follow these steps:

1. Audio Representation:
   - Extract the audio from the video using ffmpeg.
   - Calculate 24 Mel Frequency Cepstral Coefficients (MFCCs) from the audio using librosa.
   - Standardize the MFCCs sample-wise by making them zero-mean and with unit variance.
   - Use pre-padding with 0 to unify the length of the samples.
   - The audio representation per sample will be a tensor with shape (N, M, 1), where N is the number of coefficients (e.g. 24), and M is the number of audio frames.

2. Visual Representation:
   - Extract the frames from the video using ffmpeg.
   - Resize the images to 140x248 to preserve the aspect ratio (other resolutions can be used as well).
   - Subsample the frames to reduce complexity (e.g., 6 frames per video).
   - Use random crop (128x128) for training and center crop for validation and test sets.
   - Apply standard data augmentation techniques and scale the values to be between 0 and 1.
   - The video representation per sample will be a tensor with shape (F, H, W, 3), where F is the number of frames (e.g., 6), and H and W are spatial dimensions (e.g., 128).

3.  Transcription:
   - Takes a transcript file name and a boolean flag indicating if it's used for training. 
   - Calculates various features from the transcript, including the total number of words, the number of unique words, and the ratio of total words to unique words for each key in the transcript.
   - Counts the occurrence of filler words, although this feature is commented out and considered optional.
   - Returns a dictionary with each key representing a unique identifier from the transcript and associated values containing the calculated features.
     
4. Ground Truth Labels:
   - There are 5 targets (personality traits). Plot the distributions of these traits.
   - Be aware of the 'regression-to-the-mean problem.'

5. Create a Generator:
   - Implement a generator that iterates over the audio and visual representations and produces a tuple ([x0, x1], y), where x0 is the audio, x1 is the video representation, x2 is the transcript representation and y is the ground truth (a 5x1 vector for each sample).

6. Model Creation:
   - Create the audio subnetwork and choose from one of the suggested configurations (BLSTM, Conv1D, or Conv2D).
   - Create the visual subnetwork using a visual backbone, such as VGG-like architecture or ResNet50/Inception architecture.
   - Create the transcription subnetwork using word embedding and bidirectional LSTM.
   - Concatenate the final hidden representations of the audio, visual and transcription subnetworks.
   - Apply fully connected layers (256 units, ReLU), followed by another dense layer (5 units, linear or sigmoid).
   - You can feed multiple inputs to the Model using a list.

7. Performance Metric:
   - Use the 1-Mean Absolute Error (1-MAE) as the evaluation metric.

8. Final Evaluation:
   - Plot the training/validation curves and the '1-MAE' performance metric.
   - Calculate the coefficient of determination (R^2) regression metric on the train, validation, and test sets after training. Note that monitoring this metric during training with small batch sizes is not advised due to noise and misleading results.
