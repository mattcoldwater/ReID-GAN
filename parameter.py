import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='bigbigan', choices=['bigbigan', 'qgan'])
    parser.add_argument('--adv_loss', type=str, default='hinge', choices=['wgan-gp', 'hinge'])
    parser.add_argument('--g_num', type=int, default=5)
    parser.add_argument('--chn', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=120)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--version', type=str, default='biggan_market1501')
    parser.add_argument('--n_class', type=int, default=751)
    parser.add_argument('--trans', type=int, default=1, help='0:mask, 1:bouding box')

    # Training setting
    parser.add_argument('--total_step', type=int, default=201, help='how many times to update the generator') # 1000000
    parser.add_argument('--d_iters', type=float, default=5)
    # parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batchid', type=int, default=2)
    parser.add_argument('--batchimage', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2) #12
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=100)

    # Testing setting
    parser.add_argument('--batchtest', type=int, default=4) #64    

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--gpus', type=str, default='0', help='gpuids eg: 0,1,2,3  --parallel True  ')
    parser.add_argument('--dataset', type=str, default='market1501')
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--image_path', type=str, default='../market1501/Market1501') # ../tiny-imagenet-200/train
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')

    # Step size
    parser.add_argument('--log_step', type=int, default=100) # 10
    parser.add_argument('--sample_step', type=int, default=100) #100
    parser.add_argument('--model_save_step', type=float, default=10.0) #1.0


    return parser.parse_args()
