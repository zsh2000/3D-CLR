_base_ = './default_ubd_inward_facing.py'

expname = '00109-GTV2Y73Sn5t_12_10'

data = dict(
    datadir='00109-GTV2Y73Sn5t_12_10',
    factor=2,
    movie_render_kwargs={
        'scale_r': 1.0,
        'scale_f': 0.8,
        'zrate': 2.0,
        'zdelta': 0.5,
    }
)

fine_train = dict(
    N_iters=100000,
)
