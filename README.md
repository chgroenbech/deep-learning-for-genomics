# Deep Learning for Genomics #

Project for special course at DTU Compute together with [Maximillian Fornitz Vording][Max] in collaboration with [Pers Lab][].

[Max]: https://github.com/maximillian91
[Pers Lab]: https://github.com/perslab

Using deep learning methods on single-cell sequencing data, we want cluster cells  and potentially classify their cell types.

## Setup ##

The model has been implemented in Python using the [Theano][], [Lasagne][], and [Parmesan][] modules. In addition, [NumPy][] and [matplotlib][] are also used to for computations and making figures.

[Theano]: http://deeplearning.net/software/theano/
[Lasagne]: http://lasagne.readthedocs.io/en/latest/index.html
[Parmesan]: https://github.com/casperkaae/parmesan
[NumPy]: http://www.numpy.org
[matplotlib]: http://matplotlib.org

Data are expected to be in subdirectory called `data`. All other necessary subdirectories are created as needed.

## Running ##

For how to run the model, run `./src/main.py -h`.

The shell script `run.sh` has been supplied to make it easier to specify the arguments to `./src/main.py`.

## Credits ##

Most of the model specification has been taken from [implementation of the variational auto-encoder][VAE] from the [Deep Learning course][deep-learning] at DTU.

[VAE]: https://github.com/DeepLearningDTU/02456-deep-learning/blob/master/week5/lab52_VAE-lasagne.ipynb
[deep-learning]: https://github.com/DeepLearningDTU/02456-deep-learning
