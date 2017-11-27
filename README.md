 ## Objective

The recent state of the art Artificial Intelligence (AI) can recognize images and knows how to read and listen. To add onto that, it would be really amazing if machines can not only recognize music, but also create a different genre of music by listening to one particular genre of music. As AI agents become more advanced, the capability of AI being able to produce music would also be pivotal.

The goal of this project is to apply style transfer related deep learning models to transform music from one style to another (e.g. genre to genre). In order to achieve the goal, we have implemented several deep neural network architectures, such as CycleGAN and LSTM, which take the input of audio of one domain and produce the audio of another domain.

 <img src="./giphy-guitar.gif" style="margin-bottom: 115px; float: left" width="120"/> <img src="./DeepNeuralNet.jpg"  style="PADDING-LEFT: 50px; PADDING-RIGHT: 50px; float: left" width="525"/> <img src="./giphy-saxophone.gif" width="140" style="PADDING-TOP: 65px; float: left;"/><p style="clear: both;"> 


## Previous Studies
Several studies have been conducted in the area of music style transfer and generally music generation using deep learning. Some of them are listed below.


*	[Google Magenta](https://magenta.tensorflow.org/) is a Google Brain project that generates melodies using neural networks. 
*	[Sturm et al.](https://arxiv.org/pdf/1604.08723.pdf) utilized a character-based model with an LSTM to generate a textual representation of a song for music transcription modelling and composition applications.
*	[Johnson](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/) developed a biaxial RNN model, consists of time axis and the note axis, to compose music. 
*	[Malik and Ek](https://arxiv.org/pdf/1708.03535.pdf) generated stylized performances of music, that focuses more on different music dynamics, using Bi-directional LSTM architectures.
*	[Yang et al.](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/226_Paper.pdf) proposed MIDINET framework, which uses CNNs to produce melodies that are represented by series of symbolic-domain MIDI notes.  
*	[Dong et al.](https://arxiv.org/pdf/1709.06298.pdf) introduced MuseGAN architecture, which applies GANs idea in symbolic-domain multi-track sequential music generation using a large collection of MIDIs in an unsupervised learning approach.
*	[Ulyanov and Lebedev](https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/) implemented audio style transfer via artistic style transfer algorithm, which uses convolutions with random weights to represent audio features. 
*	[Finally, Briot et al.](https://arxiv.org/pdf/1709.01620.pdf) published a survey book that discusses various ways of using deep artificial neural networks for musical content generation.

## Approach I

In the first approach ...
<p align="center">
 <img src="./CycleGAN.jpg" width="620"/>
</p>

## Approach II

In the second approach ...
<p align="center">
 <img src="./Seq2Seq.jpg" width="620"/>
</p>
