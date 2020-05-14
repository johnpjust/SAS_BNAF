import os
import json
import pprint
import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bnaf import *
import glob
import random
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b

import functools
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class parser_:
    pass

args = parser_()
args.device = '/gpu:0'  # '/gpu:0'
args.dataset = 'SAS2020'  # 'gq_ms_wheat_johnson'#'gq_ms_wheat_johnson' #['gas', 'bsds300', 'hepmass', 'miniboone', 'power']
args.learning_rate = np.float32(1e-2)
args.batch_dim = 2000
args.clip_norm = 0.1
args.epochs = 5000
args.patience = 100
args.decay = 0.5
args.flows = 6
args.layers = 1
args.hidden_dim = 12
args.residual = 'gated'
args.tensorboard = r'J:\SaS\2020\tensorboard'
args.crop_size = np.array([5, 5, 3])
args.n_dims = np.prod(args.crop_size)
args.spacing = 2

def create_model(args, verbose=False):
    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    tf.random.set_seed(args.manualSeedw)
    np.random.seed(args.manualSeedw)

    dtype_in = tf.float32

    g_constraint = lambda x: tf.nn.relu(x) + 1e-6  ## for batch norm
    flows = []
    for f in range(args.flows):
        # build internal layers for a single flow
        layers = []
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
                                       args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in))
            layers.append(Tanh(dtype_in=dtype_in))

        flows.append(
            BNAF(layers=[MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in),
                         Tanh(dtype_in=dtype_in)] + \
                        layers + \
                        [MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims, dtype_in=dtype_in)], \
                 res=args.residual if f < args.flows - 1 else None, dtype_in=dtype_in
                 )
        )
        ## with batch norm example
        # for _ in range(args.layers - 1):
        #     layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
        #                                args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in))
        #     layers.append(CustomBatchnorm(gamma_constraint = g_constraint, momentum=args.momentum))
        #     layers.append(Tanh(dtype_in=dtype_in))
        #
        # flows.append(
        #     BNAF(layers = [MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims, dtype_in=dtype_in), CustomBatchnorm(gamma_constraint = g_constraint, momentum=args.momentum), Tanh(dtype_in=dtype_in)] + \
        #        layers + \
        #        [CustomBatchnorm(scale=False, momentum=args.momentum), MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims, dtype_in=dtype_in)], \
        #      res=args.residual if f < args.flows - 1 else None, dtype_in= dtype_in
        #      )
        # )

        if f < args.flows - 1:
            flows.append(Permutation(args.n_dims, 'flip'))

        model = Sequential(flows)  # , dtype_in=dtype_in)
        # params = np.sum(np.sum(p.numpy() != 0) if len(p.numpy().shape) > 1 else p.numpy().shape
        #              for p in model.trainable_variables)[0]

    return model

model = create_model(args)
model.load_weights(r'')

def img_heatmap(img_crop):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img_crop, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)

    rows = img.shape[0] - args.crop_size[0]
    cols = img.shape[1] - args.crop_size[1]
    heatmap = np.zeros((np.int(rows / args.spacing)+1, np.int(cols / args.spacing)+1, model.output_shape[-1]))
    im_breakup_array = np.zeros((np.int(cols / args.spacing)+1, *args.crop_size), dtype=np.float32)
    with tf.device(args.device):
        for i in range(0, rows+1, args.spacing):
            for j in range(0, cols+1, args.spacing):
                im_breakup_array[np.int(j / args.spacing), :] = tf.image.crop_to_bounding_box(img, i, j, args.crop_size[0], args.crop_size[1])
            heatmap[np.int(i / args.spacing), :] = model(im_breakup_array, training=False).numpy()
    return heatmap


def load_model(args, root, load_start_epoch=False):
    # def f():
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load))
    # root.restore(os.path.join(args.load or args.path, 'checkpoint'))
    # if load_start_epoch:
    #     args.start_epoch = tf.train.get_global_step().numpy()
    # return f

# @tf.function
def compute_log_p_x(model, x_mb):
    ## use tf.gradient + tf.convert_to_tensor + tf.GradientTape(persistent=True) to clean up garbage implementation in bnaf.py
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = tf.reduce_sum(tfp.distributions.Normal(tf.zeros_like(y_mb), tf.ones_like(y_mb)).log_prob(y_mb),
                               axis=-1)  # .sum(-1)
    return log_p_y_mb + log_diag_j_mb


class parser_:
    pass

def main():
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    # tf.compat.v1.enable_eager_execution(config=config)

    # tf.config.experimental_run_functions_eagerly(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    print('Loading dataset..')
    trainval = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in trainval])
    cont_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    cont_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in cont_data])

    if args.vh:
        args.vh = np.load(args.path + '/vh.npy', args.vh)

    args.rand_box_size = np.int(train_data[0].shape[0] * args.rand_box_init)
    args.rand_box = np.array([args.rand_box_size, args.rand_box_size, 3])
    args.n_dims = np.prod(args.rand_box)

    print('Creating BNAF model..')
    with tf.device(args.device):
        model = create_model(args, verbose=True)

    root = None
    print('Creating optimizer..')
    with tf.device(args.device):
        optimizer = tf.optimizers.Adam()
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.compat.v1.train.get_or_create_global_step())

    if args.load:
        load_model(args, root, load_start_epoch=True)

    heat_map_func = functools.partial(compute_log_p_x, model=model)

    heat_map = []
    heat_map.extend(img_heatmap(heat_map_func, f, args) for f in fnames)
    heatmap_ = np.array(heat_map)

    ## johnsonsu xfrm for density fit (5.4155884341570175, 4.78009012658631, 622.0617883438022, 214.5187927541507)
    # dist = [5.4155884341570175, 4.78009012658631, 622.0617883438022, 214.5187927541507]
    dist = (8.144493590964167, 6.017963993607797, 740.3910154966748, 219.38576508100834)
    heatmap_ = (np.arcsinh((heatmap_ - dist[-2]) / dist[-1]) * dist[1] + dist[0])



    heatmap_ = tf.sigmoid((np.arcsinh((heatmap_ - dist[-2]) / dist[-1]) * dist[1] + dist[0])).numpy()
    ##call function directly first
    i = 0
    plt.figure();plt.imshow(get_image(fnames[i], args)/255)
    plt.figure()
    plt.imshow(heatmap_[i], cmap='hot', interpolation='nearest', vmin=0, vmax=1, alpha=1)

if __name__ == '__main__':
    main()

##"C:\Program Files\Git\bin\sh.exe" --login -i

#### tensorboard --logdir=C:\Users\justjo\PycharmProjects\BNAF_tensorflow_eager\tensorboard\checkpoint
## http://localhost:6006/