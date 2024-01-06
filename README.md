# Vision Transformer
Pytorch reimplementation of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) that was released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.

This paper show that Transformers applied directly to image patches and pre-trained on large datasets work really well on image recognition task.

![fig1](./img/figure1.png)

Vision Transformer achieve State-of-the-Art in image recognition task with standard Transformer encoder and fixed-size patches. In order to perform classification, author use the standard approach of adding an extra learnable "classification token" to the sequence.

![fig2](./img/figure2.png)


## Usage
### 1. Download Pre-trained model (Google's Official Checkpoint)
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16
```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz

```

### 2. Train Model
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
```
CIFAR-10 and CIFAR-100 are automatically download and train. In order to use a different dataset you need to customize [data_utils.py](./utils/data_utils.py).

The default batch size is 512. When GPU memory is insufficient, you can proceed with training by adjusting the value of `--gradient_accumulation_steps`.

Also can use [Automatic Mixed Precision(Amp)](https://nvidia.github.io/apex/amp.html) to reduce memory usage and train faster
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
```



## Results
To verify that the converted model weight is correct, we simply compare it with the author's experimental results. We trained using mixed precision, and `--fp16_opt_level` was set to O2.

### imagenet-21k
* [**tensorboard**](https://tensorboard.dev/experiment/Oz9GmmQIQCOEr4xbdr8O3Q)

|    model     |  dataset  | resolution | acc(official) | acc(this repo) |  time   |
|:------------:|:---------:|:----------:|:-------------:|:--------------:|:-------:|
|   ViT-B_16   | CIFAR-10  |  224x224   |       -       |     0.9908     | 3h 13m  |
|   ViT-B_16   | CIFAR-10  |  384x384   |    0.9903     |     0.9906     | 12h 25m |
|   ViT_B_16   | CIFAR-100 |  224x224   |       -       |     0.923      |  3h 9m  |
|   ViT_B_16   | CIFAR-100 |  384x384   |    0.9264     |     0.9228     | 12h 31m |
| R50-ViT-B_16 | CIFAR-10  |  224x224   |       -       |     0.9892     | 4h 23m  |
| R50-ViT-B_16 | CIFAR-10  |  384x384   |     0.99      |     0.9904     | 15h 40m |
| R50-ViT-B_16 | CIFAR-100 |  224x224   |       -       |     0.9231     | 4h 18m  |
| R50-ViT-B_16 | CIFAR-100 |  384x384   |    0.9231     |     0.9197     | 15h 53m |
|   ViT_L_32   | CIFAR-10  |  224x224   |       -       |     0.9903     | 2h 11m  |
|   ViT_L_32   | CIFAR-100 |  224x224   |       -       |     0.9276     |  2h 9m  |
|   ViT_H_14   | CIFAR-100 |  224x224   |       -       |      WIP       |         |


### imagenet-21k + imagenet2012
* [**tensorboard**](https://tensorboard.dev/experiment/CXOzjFRqTM6aLCk0jNXgAw/#scalars)

|    model     |  dataset  | resolution |  acc   |
|:------------:|:---------:|:----------:|:------:|
| ViT-B_16-224 | CIFAR-10  |  224x224   |  0.99  |
| ViT_B_16-224 | CIFAR-100 |  224x224   | 0.9245 |
|   ViT-L_32   | CIFAR-10  |  224x224   | 0.9903 |
|   ViT-L_32   | CIFAR-100 |  224x224   | 0.9285 |


### shorter train
* In the experiment below, we used a resolution size (224x224).
* [**tensorboard**](https://tensorboard.dev/experiment/lpknnMpHRT2qpVrSZi10Ag/#scalars)

|  upstream   |  model   |  dataset  | total_steps /warmup_steps | acc(official) | acc(this repo) |
|:-----------:|:--------:|:---------:|:-------------------------:|:-------------:|:--------------:|
| imagenet21k | ViT-B_16 | CIFAR-10  |          500/100          |    0.9859     |     0.9859     |
| imagenet21k | ViT-B_16 | CIFAR-10  |         1000/100          |    0.9886     |     0.9878     |
| imagenet21k | ViT-B_16 | CIFAR-100 |          500/100          |    0.8917     |     0.9072     |
| imagenet21k | ViT-B_16 | CIFAR-100 |         1000/100          |    0.9115     |     0.9216     |


## Visualization
The ViT consists of a Standard Transformer Encoder, and the encoder consists of Self-Attention and MLP module.
The attention map for the input image can be visualized through the attention score of self-attention.

Visualization code can be found at [visualize_attention_map](./visualize_attention_map.ipynb).

![fig3](./img/figure3.png)


## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [Pytorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models)


## Citations

```bibtex
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

# Changes

# Avoid overriding CUDA installation

The default dev container installation configuration adds the UDa libraries and tools. The additional OS installation updates the packages. When this is done, some CUDA libraries are updated. The OS then has the incorrect libraries:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ apt list --installed | grep -i 12.2

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

dbus-user-session/jammy-updates,jammy-security,now 1.12.20-2ubuntu4.1 amd64 [installed,automatic]
dbus/jammy-updates,jammy-security,now 1.12.20-2ubuntu4.1 amd64 [installed,automatic]
libcudnn8-dev/unknown,now 8.9.7.29-1+cuda12.2 amd64 [installed]
libcudnn8/unknown,now 8.9.7.29-1+cuda12.2 amd64 [installed]
libdbus-1-3/jammy-updates,jammy-security,now 1.12.20-2ubuntu4.1 amd64 [installed,automatic]
libnode72/jammy-updates,jammy-security,now 12.22.9~dfsg-1ubuntu3.3 amd64 [installed,automatic]
libnvjpeg-12-1/unknown,now 12.2.0.2-1 amd64 [installed,automatic]
libnvjpeg-dev-12-1/unknown,now 12.2.0.2-1 amd64 [installed,automatic]
nodejs-doc/jammy-updates,jammy-security,now 12.22.9~dfsg-1ubuntu3.3 all [installed,automatic]
nodejs/jammy-updates,jammy-security,now 12.22.9~dfsg-1ubuntu3.3 amd64 [installed]
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ 
```
<!--- cSpell:enable --->

We add the [following commands](https://help.ubuntu.com/community/PinningHowto) by [pinning](https://linuxopsys.com/topics/exclude-specific-package-apt-upgrade) the packages to avoid the changes in their version:

<!--- cSpell:disable --->
```shell
sudo apt-mark hold cuda-toolkit libcudnn8-dev libcudnn8
sudo apt-get upgrade -y
```
<!--- cSpell:enable --->


Check CUDA access:

<!--- cSpell:disable --->
```shell
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python has_cuda.py 
has_cuda = True
device = cuda
n_gpu = 1
```
<!--- cSpell:enable --->


<!--- cSpell:disable --->
```shell
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
```
<!--- cSpell:enable --->

vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 16, in <module>
    from torch.utils.tensorboard import SummaryWriter
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/utils/tensorboard/__init__.py", line 2, in <module>
    from packaging.version import Version
ModuleNotFoundError: No module named 'packaging'


https://stackoverflow.com/questions/42222096/no-module-named-packaging
https://blog.finxter.com/fixed-modulenotfounderror-no-module-named-packaging/


vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 22, in <module>
    from models.modeling import VisionTransformer, CONFIGS
  File "/workspaces/ViT-pytorch/models/modeling.py", line 18, in <module>
    from scipy import ndimage
ModuleNotFoundError: No module named 'scipy'

vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 13, in <module>
    import torch.distributed as dist
ModuleNotFoundError: No module named 'torch.distributed'

Restart the dev container



vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 26, in <module>
    from utils.data_utils import get_loader
  File "/workspaces/ViT-pytorch/utils/data_utils.py", line 5, in <module>
    from torchvision import transforms, datasets
ModuleNotFoundError: No module named 'torchvision'



vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 17, in <module>
    from apex import amp
ModuleNotFoundError: No module named 'apex'

https://github.com/NVIDIA/apex/issues/1724
https://github.com/NVIDIA/apex/issues/1722
https://discuss.pytorch.org/t/how-to-replace-apex-amp-by-pytorch-amp/182087
https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

https://discuss.pytorch.org/t/torch-cuda-amp-vs-nvidia-apex/74994
# https://pytorch.org/docs/stable/notes/amp_examples.html
# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
# https://pytorch.org/docs/stable/amp.html
https://pytorch.org/docs/master/amp.html


vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
01/06/2024 12:57:22 - WARNING - __main__ - Process rank: -1, device: cuda, n_gpu: 1, distributed training: False, 16-bits training: False
Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 332, in <module>
    main()
  File "/workspaces/ViT-pytorch/train.py", line 325, in main
    args, model = setup(args)
  File "/workspaces/ViT-pytorch/train.py", line 69, in setup
    model.load_from(np.load(args.pretrained_dir))
  File "/home/vscode/.local/lib/python3.10/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoint/ViT-B_16.npz'
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ 


* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16
```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
cd checkpoint/
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz


# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz


vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
01/06/2024 13:00:08 - WARNING - __main__ - Process rank: -1, device: cuda, n_gpu: 1, distributed training: False, 16-bits training: False
01/06/2024 13:00:10 - INFO - __main__ - classifier: token
hidden_size: 768
patches:
  size: !!python/tuple
  - 16
  - 16
representation_size: null
transformer:
  attention_dropout_rate: 0.0
  dropout_rate: 0.1
  mlp_dim: 3072
  num_heads: 12
  num_layers: 12

01/06/2024 13:00:10 - INFO - __main__ - Training parameters Namespace(name='cifar10-100_500', dataset='cifar10', model_type='ViT-B_16', pretrained_dir='checkpoint/ViT-B_16.npz', output_dir='output', img_size=224, train_batch_size=512, eval_batch_size=64, eval_every=100, learning_rate=0.03, weight_decay=0, num_steps=10000, decay_type='cosine', warmup_steps=500, max_grad_norm=1.0, local_rank=-1, seed=42, gradient_accumulation_steps=1, fp16=False, fp16_opt_level='O2', loss_scale=0, n_gpu=1, device=device(type='cuda'))
01/06/2024 13:00:10 - INFO - __main__ - Total Parameter:        85.8M
85.806346
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 170498071/170498071 [00:07<00:00, 23253604.21it/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
01/06/2024 13:00:21 - INFO - __main__ - ***** Running training *****
01/06/2024 13:00:21 - INFO - __main__ -   Total optimization steps = 10000
01/06/2024 13:00:21 - INFO - __main__ -   Instantaneous batch size per GPU = 512
01/06/2024 13:00:21 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 512
01/06/2024 13:00:21 - INFO - __main__ -   Gradient Accumulation steps = 1
Training (X / X Steps) (loss=X.X):   0%|| 0/98 [00:00<?, ?it/s]ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
Training (X / X Steps) (loss=X.X):   0%|| 0/98 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1132, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/usr/lib/python3.10/queue.py", line 180, in get
    self.not_empty.wait(remaining)
  File "/usr/lib/python3.10/threading.py", line 324, in wait
    gotit = waiter.acquire(True, timeout)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 13068) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 332, in <module>
    main()
  File "/workspaces/ViT-pytorch/train.py", line 328, in main
    train(args, model)
  File "/workspaces/ViT-pytorch/train.py", line 197, in train
    for step, batch in enumerate(epoch_iterator):
  File "/home/vscode/.local/lib/python3.10/site-packages/tqdm/std.py", line 1182, in __iter__
    for obj in iterable:
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1328, in _next_data
    idx, data = self._get_data()
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1284, in _get_data
    success, data = self._try_get_data()
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1145, in _try_get_data
    raise RuntimeError(f'DataLoader worker (pid(s) {pids_str}) exited unexpectedly') from e
RuntimeError: DataLoader worker (pid(s) 13068) exited unexpectedly
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ 


https://discuss.pytorch.org/t/dataset-size-and-limited-shared-memory/171135
https://discuss.pytorch.org/t/shared-memory-with-torch-multiprocessing/87921
https://community.databricks.com/t5/data-engineering/how-can-the-shared-memory-size-dev-shm-be-increased-on/td-p/12207
"PyTorch uses shared memory to efficiently share tensors between its dataloader workers and its main process. However in a docker container the default size of the shared memory (a tmpfs file system mounted at /dev/shm) is 64MB, which is too small to use to share image tensor batches. This means that when using a custom docker image on a databricks cluster it is not possible to use PyTorch with multiple dataloaders. We can fix this by setting the `--shm-size` or `--ipc=host` args on `docker run` - how can this be set on a databricks cluster?

Note that this doesn't affect the default databricks runtime it looks like that is using the linux default of making half the physical RAM available to /dev/shm - 6.9GB on the Standard_DS3_v2 node I tested.

To reproduce: start a cluster using a custom docker image, run `df -h /dev/shm` in a notebook."

```shell
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ df -h /dev/shm
Filesystem      Size  Used Avail Use% Mounted on
shm              64M     0   64M   0% /dev/shm
```

https://github.com/microsoft/vscode-remote-release/issues/3462
"Not sure this applies, but docker run has a --shm-size option. You can add that with "runArgs" in your devcontainer.json."
https://stackoverflow.com/questions/58521691/how-can-i-set-a-memory-limit-for-a-docker-container-created-by-visual-studio-cod
reference https://datawookie.dev/blog/2021/11/shared-memory-docker/

https://askubuntu.com/questions/898941/how-to-check-ram-size

vscode ➜ /workspaces/ViT-pytorch (dev_container) $ free -m
               total        used        free      shared  buff/cache   available
Mem:          112682        4997       66757           4       40926      106580
Swap:              0           0           0

or --mega or -g 

vscode ➜ /workspaces/ViT-pytorch (dev_container) $ free -h --si 
               total        used        free      shared  buff/cache   available
Mem:            112G        5.0G         66G        4.0M         40G        106G
Swap:             0B          0B          0B


vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
01/06/2024 13:33:01 - WARNING - __main__ - Process rank: -1, device: cuda, n_gpu: 1, distributed training: False, 16-bits training: False
01/06/2024 13:33:03 - INFO - __main__ - classifier: token
hidden_size: 768
patches:
  size: !!python/tuple
  - 16
  - 16
representation_size: null
transformer:
  attention_dropout_rate: 0.0
  dropout_rate: 0.1
  mlp_dim: 3072
  num_heads: 12
  num_layers: 12

01/06/2024 13:33:03 - INFO - __main__ - Training parameters Namespace(name='cifar10-100_500', dataset='cifar10', model_type='ViT-B_16', pretrained_dir='checkpoint/ViT-B_16.npz', output_dir='output', img_size=224, train_batch_size=512, eval_batch_size=64, eval_every=100, learning_rate=0.03, weight_decay=0, num_steps=10000, decay_type='cosine', warmup_steps=500, max_grad_norm=1.0, local_rank=-1, seed=42, gradient_accumulation_steps=1, fp16=False, fp16_opt_level='O2', loss_scale=0, n_gpu=1, device=device(type='cuda'))
01/06/2024 13:33:03 - INFO - __main__ - Total Parameter:        85.8M
85.806346
Files already downloaded and verified
Files already downloaded and verified
01/06/2024 13:33:04 - INFO - __main__ - ***** Running training *****
01/06/2024 13:33:04 - INFO - __main__ -   Total optimization steps = 10000
01/06/2024 13:33:04 - INFO - __main__ -   Instantaneous batch size per GPU = 512
01/06/2024 13:33:04 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 512
01/06/2024 13:33:04 - INFO - __main__ -   Gradient Accumulation steps = 1
Training (X / X Steps) (loss=X.X):   0%|| 0/98 [00:02<?, ?it/s]
Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 332, in <module>
    main()
  File "/workspaces/ViT-pytorch/train.py", line 328, in main
    train(args, model)
  File "/workspaces/ViT-pytorch/train.py", line 200, in train
    loss = model(x, y)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspaces/ViT-pytorch/models/modeling.py", line 273, in forward
    x, attn_weights = self.transformer(x)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspaces/ViT-pytorch/models/modeling.py", line 258, in forward
    encoded, attn_weights = self.encoder(embedding_output)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspaces/ViT-pytorch/models/modeling.py", line 243, in forward
    hidden_states, weights = layer_block(hidden_states)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspaces/ViT-pytorch/models/modeling.py", line 188, in forward
    x = self.ffn(x)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspaces/ViT-pytorch/models/modeling.py", line 116, in forward
    x = self.fc1(x)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.15 GiB. GPU 0 has a total capacty of 23.69 GiB of which 912.81 MiB is free. Process 454738 has 22.79 GiB memory in use. Of the allocated memory 21.50 GiB is allocated by PyTorch, and 1001.89 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --gradient_accumulation_steps 2

vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
01/06/2024 13:37:18 - WARNING - __main__ - Process rank: -1, device: cuda, n_gpu: 1, distributed training: False, 16-bits training: True
01/06/2024 13:37:19 - INFO - __main__ - classifier: token
hidden_size: 768
patches:
  size: !!python/tuple
  - 16
  - 16
representation_size: null
transformer:
  attention_dropout_rate: 0.0
  dropout_rate: 0.1
  mlp_dim: 3072
  num_heads: 12
  num_layers: 12

01/06/2024 13:37:19 - INFO - __main__ - Training parameters Namespace(name='cifar10-100_500', dataset='cifar10', model_type='ViT-B_16', pretrained_dir='checkpoint/ViT-B_16.npz', output_dir='output', img_size=224, train_batch_size=512, eval_batch_size=64, eval_every=100, learning_rate=0.03, weight_decay=0, num_steps=10000, decay_type='cosine', warmup_steps=500, max_grad_norm=1.0, local_rank=-1, seed=42, gradient_accumulation_steps=1, fp16=True, fp16_opt_level='O2', loss_scale=0, n_gpu=1, device=device(type='cuda'))
01/06/2024 13:37:19 - INFO - __main__ - Total Parameter:        85.8M
85.806346
Files already downloaded and verified
Files already downloaded and verified
Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 332, in <module>
    main()
  File "/workspaces/ViT-pytorch/train.py", line 328, in main
    train(args, model)
  File "/workspaces/ViT-pytorch/train.py", line 168, in train
    model, optimizer = amp.initialize(models=model,
AttributeError: module 'torch.cuda.amp' has no attribute 'initialize'
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --gradient_accumulation_steps 2
01/06/2024 13:44:49 - WARNING - __main__ - Process rank: -1, device: cuda, n_gpu: 1, distributed training: False, 16-bits training: False
01/06/2024 13:44:51 - INFO - __main__ - classifier: token
hidden_size: 768
patches:
  size: !!python/tuple
  - 16
  - 16
representation_size: null
transformer:
  attention_dropout_rate: 0.0
  dropout_rate: 0.1
  mlp_dim: 3072
  num_heads: 12
  num_layers: 12

01/06/2024 13:44:51 - INFO - __main__ - Training parameters Namespace(name='cifar10-100_500', dataset='cifar10', model_type='ViT-B_16', pretrained_dir='checkpoint/ViT-B_16.npz', output_dir='output', img_size=224, train_batch_size=512, eval_batch_size=64, eval_every=100, learning_rate=0.03, weight_decay=0, num_steps=10000, decay_type='cosine', warmup_steps=500, max_grad_norm=1.0, local_rank=-1, seed=42, gradient_accumulation_steps=2, fp16=False, fp16_opt_level='O2', loss_scale=0, n_gpu=1, device=device(type='cuda'))
01/06/2024 13:44:51 - INFO - __main__ - Total Parameter:        85.8M
85.806346
Files already downloaded and verified
Files already downloaded and verified
01/06/2024 13:44:52 - INFO - __main__ - ***** Running training *****
01/06/2024 13:44:52 - INFO - __main__ -   Total optimization steps = 10000
01/06/2024 13:44:52 - INFO - __main__ -   Instantaneous batch size per GPU = 256
01/06/2024 13:44:52 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 512
01/06/2024 13:44:52 - INFO - __main__ -   Gradient Accumulation steps = 2
Training (X / X Steps) (loss=X.X):   0%|| 0/196 [00:01<?, ?it/s]
Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 332, in <module>
    main()
  File "/workspaces/ViT-pytorch/train.py", line 328, in main
    train(args, model)
  File "/workspaces/ViT-pytorch/train.py", line 200, in train
    loss = model(x, y)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspaces/ViT-pytorch/models/modeling.py", line 273, in forward
    x, attn_weights = self.transformer(x)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspaces/ViT-pytorch/models/modeling.py", line 258, in forward
    encoded, attn_weights = self.encoder(embedding_output)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspaces/ViT-pytorch/models/modeling.py", line 243, in forward
    hidden_states, weights = layer_block(hidden_states)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspaces/ViT-pytorch/models/modeling.py", line 183, in forward
    x, weights = self.attn(x)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/workspaces/ViT-pytorch/models/modeling.py", line 90, in forward
    context_layer = torch.matmul(attention_probs, value_layer)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 148.00 MiB. GPU 0 has a total capacty of 23.69 GiB of which 76.81 MiB is free. Process 457819 has 23.60 GiB memory in use. Of the allocated memory 23.11 GiB is allocated by PyTorch, and 184.89 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ 

torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.15 GiB. GPU 0 has a total capacty of 23.69 GiB of which 912.81 MiB is free. Process 454738 has 22.79 GiB memory in use. Of the allocated memory 21.50 GiB is allocated by PyTorch, and 1001.89 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
01/06/2024 13:56:17 - WARNING - __main__ - Process rank: -1, device: cuda, n_gpu: 1, distributed training: False, 16-bits training: True
01/06/2024 13:56:19 - INFO - __main__ - classifier: token
hidden_size: 768
patches:
  size: !!python/tuple
  - 16
  - 16
representation_size: null
transformer:
  attention_dropout_rate: 0.0
  dropout_rate: 0.1
  mlp_dim: 3072
  num_heads: 12
  num_layers: 12

01/06/2024 13:56:19 - INFO - __main__ - Training parameters Namespace(name='cifar10-100_500', dataset='cifar10', model_type='ViT-B_16', pretrained_dir='checkpoint/ViT-B_16.npz', output_dir='output', img_size=224, train_batch_size=512, eval_batch_size=64, eval_every=100, learning_rate=0.03, weight_decay=0, num_steps=10000, decay_type='cosine', warmup_steps=500, max_grad_norm=1.0, local_rank=-1, seed=42, gradient_accumulation_steps=1, fp16=True, fp16_opt_level='O2', loss_scale=0, n_gpu=1, device=device(type='cuda'))
01/06/2024 13:56:19 - INFO - __main__ - Total Parameter:        85.8M
85.806346
Files already downloaded and verified
Files already downloaded and verified
Traceback (most recent call last):
  File "/workspaces/ViT-pytorch/train.py", line 332, in <module>
    main()
  File "/workspaces/ViT-pytorch/train.py", line 328, in main
    train(args, model)
  File "/workspaces/ViT-pytorch/train.py", line 168, in train
    model, optimizer = amp.initialize(models=model,
AttributeError: module 'torch.cuda.amp' has no attribute 'initialize'
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ 
