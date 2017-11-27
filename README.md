 ## Objective

The recent state of the art Artificial Intelligence (AI) can recognize images and knows how to read and listen. To add onto that, it would be really amazing if machines can not only recognize music, but also create a different genre of music by listening to one particular genre of music. As AI agents become more advanced, the capability of AI being able to produce music would also be pivotal.

The goal of this project is to apply style transfer related deep learning models to transform music from one style to another (e.g. genre to genre). In order to achieve the goal, we have implemented several deep neural network architectures, such as CycleGAN and LSTM, which take the input of audio of one domain and produce the audio of another domain.
<p align="center">
<img src="./giphy-guitar.gif" style="margin-bottom: 115px; float: left" width="140"/> <img src="./DeepNeuralNet.jpg"  style="PADDING-LEFT: 50px; PADDING-RIGHT: 50px; float: left" width="455"/> <img src="./giphy-saxophone.gif" width="170" style="PADDING-TOP: 65px; float: left;"/><p style="clear: both;"> 
</p>


## Previous Studies
Several studies have been conducted in the area of music style transfer and generally music generation using deep learning. Some of them are listed below.


*	[Google Magenta](https://magenta.tensorflow.org/) is a Google Brain project that generates melodies using neural networks. 
*	[Sturm et al.](https://arxiv.org/pdf/1604.08723.pdf) utilized a character-based model with an LSTM to generate a textual representation of a song for music transcription modelling and composition applications.
*	[Johnson](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/) developed a biaxial RNN model, consists of time axis and the note axis, to compose music. 
*	[Malik and Ek](https://arxiv.org/pdf/1708.03535.pdf) generated stylized performances of music, that focuses more on different music dynamics, using Bi-directional LSTM architectures.
*	[Yang et al.](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/226_Paper.pdf) proposed MIDINET framework, which uses CNNs to produce melodies that are represented by series of symbolic-domain MIDI notes.  
*	[Dong et al.](https://arxiv.org/pdf/1709.06298.pdf) introduced MuseGAN architecture, which applies GANs idea in symbolic-domain multi-track sequential music generation using a large collection of MIDIs in an unsupervised learning approach.
*	[Ulyanov and Lebedev](https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/) implemented audio style transfer via artistic style transfer algorithm, which uses convolutions with random weights to represent audio features. 
*	Finally, [Briot et al.](https://arxiv.org/pdf/1709.01620.pdf) published a survey book that discusses various ways of using deep artificial neural networks for musical content generation.

## Approach I (CycleGAN- Mark's results)

In the first approach ...
<p align="center">
 <img src="./CycleGAN.jpg" width="720"/>
</p>

## Approach II (Seq2Seq- Likith's results)

Machine Translation has had a great success by utilizing Seq2Seq models which use LSTM networks to model general purpose encoder-decoder frameworks. For machine translation, Seq2Seq models take text in one language as input and produce an intermediate vector which is further used to produce text in the target language. We applied the same technique to translate between instruments of a midi file, which would achieve the objective of style transfer as music can vary between instruments in the way they are played.

<p align="center">
 <img src="./Seq2Seq.jpg" width="720"/>
</p>

For this experiment, MIDI data from Clean MIDI subset within Lakh MIDI dataset was used. We considered two instruments that co-occur most frequently in each file in the dataset - Grand Acoustic Piano and String Ensemble. For each file containing both instruments, notes were extracted separately. No timing information was retained. These note numbers varied between 0 and 128. Based on this, the input to the model was a sequence of notes corresponding to Piano and the target was a sequence of notes corresponding to String Ensemble of a particular MIDI file. There were around 300 tracks with the instruments Grand Acoustic Piano (with program number 0) and String Ensemble (with program number 48) in the dataset.

The Seq2Seq model was built with a decoder with Attention mechanism. Luong attention mechanism was used. The hidden size in both encoder and decoder was 30. The embedding size of both encoder and decoder was 30. Attention size was also 30. Each unit in the encoder and decoder RNNs was an LSTM cell. The model was trained for 800 epochs with RMSprop optimizer and a learning rate of 0.001.

Style transfer achieved through this model via machine translation was not very encouraging. The notes produced as a result of translation were repeating and were off form the target sequence. The error through all the iterations did not reduce considerably. As a result, the sounds produced were not indicative of the expected output.

This approach might not have produced good results because the dataset was probably not big enough. After filtering through the dataset to find tracks with two most co-occurring instruments, only 300 tracks were found. Also, in many of these tracks, sequences corresponding to the instruments were not aligned parallely with respect to time i.e., the two instruments did not occur throughout the track and they did not occur concurrently.

“I have not failed. I've just found 10,000 ways that won't work.” - Thomas A. Edison

“We have not failed. We've just found 5 ways that won't work.” - Deep Musicians


## Approach III (CharRNN- Peter's results)

In the third approach ...

