# l3embedding
This is an implementation of the proposed model in Look, Listen and Learn ([Arandjelović, R., Zisserman, A. 2017](https://arxiv.org/pdf/1705.08168.pdf)). This model uses videos to learn vision and audio features in an unsupervised fashion by training the model for the proposed Audio-Visual Correspondence (AVC) task. This task tries to determine whether a piece of audio and an image frame come from the same video and occur simulatneously.

Dependencies
* Python 3 (we use 3.6.3)
* [ffmpeg](http://www.ffmpeg.org)
* [sox](http://sox.sourceforge.net)
* [TensorFlow](https://www.tensorflow.org/install/) (follow instructions carefully, and install before other Python dependencies)
* [keras](https://keras.io/#installation) (follow instructions carefully!)
* Other Python dependencies can by installed via `pip install -r requirements.txt`

The code for the model and training implementation can be found in `l3embedding/`. Note that the metadata format expected is the same used in [AudioSet](https://research.google.com/audioset/download.html) ([Gemmeke, J., Ellis, D., et al. 2017](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45857.pdf)), as training this model on AudioSet was one of the goals for this implementation.

You can train an AVC/embedding model using `train.py`. Run `python train.py -h` to read the help message regarding how to use the script.

There is also a module `classifier/` which contains code to train a classifier using that uses extracts embeddings on new audio using the embedding model. Currently this only supports using the [UrbanSound8K dataset](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html) ([Salamon, J., Jacoby, C., Bello, J. 2014](https://serv.cusp.nyu.edu/projects/urbansounddataset/salamon_urbansound_acmmm14.pdf))

You can train an urban sound classification model using `train_classifier.py`. Run `python train_classifier.py -h` to read the help message regarding how to use the script.


## Download VGGish models:
* `cd ./resources/vggish`
* `curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt`
* `curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz`
* `cd ../..`


If you use a SLURM environment, `sbatch` scripts are available in `jobs/`.
