_base_ = './default_ubd_inward_facing.py'

expname = '00009-vLpv2VX547B_0'

data = dict(
    datadir='00009-vLpv2VX547B_0',
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
