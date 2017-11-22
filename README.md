This is an implementation of U-Net for vocal separation proposed at ISMIR 2017, with Chainer framework.

## Requirements

Python 3.5.x

Chainer 3.0

librosa 5.0

cupy 2.0 (required if you want to train U-Net yourself. CUDA environment required.)

## Usage

Please refer to `DoExperiment.py` for code examples (or simply modify it!).

## How to prepare dataset for U-Net training

*If you want to train U-Net with your own dataset, prepare the mixed, instrumental-only, and vocal-only versions of each track, and pickle their spectrograms using `util.SaveSpectrogram()` function. You should set `PATH_FFT (in const.py)` to the directory you want to save the pickled data.

*If you have either iKala, MedleyDB, DSD100 dataset, you could make use of `ProcessXX.py` scripts. Remember to set the `PATH_XX` in each script to the right path.

*If you want to generate dataset with "original" and "instrumental version" audio pairs (as the original work did), refer to `ProcessIMAS.py`.

## Reference

The neural network is implemented according to the following publication:

*Andreas Jansson, Eric J. Humphrey, Nicola Montecchio, Rachel Bittner, Aparna Kumar, Tillman Weyde*, Singing Voice Separation with Deep U-Net Convolutional Networks, Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR), 2017.
