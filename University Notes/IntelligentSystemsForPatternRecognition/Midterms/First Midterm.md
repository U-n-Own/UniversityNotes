[[Signal Processing 1]], [[Music]], [[Assignment]]

#### From list of **Signal processing assignments**

#Assignment 

The musical pitch of a note is determined by its fundamental frequency. The pitch played by different instruments sounds different due to harmonics, i.e. other frequencies that are superimposed and determine the timbre of the instrument. [This dataset](https://philharmonia.co.uk/resources/sound-samples/) contains samples from several instruments playing different notes. Plot the spectrogram for some of them (4 instruments are sufficient) and check if it is possible to recognize the different instruments by only looking at the spectrogram. In your presentation, discuss which samples you chose to compare, how you computed the spectrogram and whether the resulting features are sufficient to recognize the instrument.

# Signal processing approaches

As we saw at lesson in [[Lesson 1 - Time series and brief Fourier analysis]], some techniques like [[Fourier Transforms]], or in [[Lesson 4 - Wavelets]] we've seen how we can analyze signal in both time and frequency domain. That's what we are intrested when we want to analyze music signal, because time in which the different harmonics with different frequencies occours matter.

# Timbre and Overtones

Basically every frequency that is a multiple of our fundamental frequency $f_0$, all the harmonics that comes are multiples of our fundamental but following a certain serie: the ***Overtone Series*** famously known as [Harmonic Series](https://en.wikipedia.org/wiki/Harmonic_series_(music)). 

Whne an instrument plays a certain note, we as humans are conditioned to hear only the fundamental, because the fundamental determines the *pitch*. 

Each instruments has a different ***Timbre***, that is with some simplification how the sounds changes overtime, so we can see that different instruments have the peak in frequency at the fundamental $f0$, and then the remaining overtones differs a lot between instruments.  


![[Pasted image 20230313142817.png]]

# Differences between instruments