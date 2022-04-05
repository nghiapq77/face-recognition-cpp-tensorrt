import argparse
import torch
# from data import cfg_mnet, cfg_slim, cfg_rfb
from config import cfg_mnet, cfg_slim, cfg_rfb
from torch2trt_dynamic import torch2trt_dynamic

# from models.retinaface import RetinaFace
from models.retinaface_trim import RetinaFace  # this version remove landmark head
from models.net_slim import Slim
from models.net_rfb import RFB


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='../../weight/torch/retina/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or slim or RFB')
# parser.add_argument('--long_side', default=320, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--width', type=int, default=320)
parser.add_argument('--height', type=int, default=288)
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg = cfg, phase = 'test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg = cfg, phase = 'test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg = cfg, phase = 'test')
    else:
        print("Don't support network!")
        exit(0)

    # load weight
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # export
    inputs = torch.randn(1, 3, args.height, args.width).to(device)
    input_names = ["input_det"]
    output_names = ["output_det0", "output_det1"]
    # convert to TensorRT feeding sample data as input
    opt_shape_param = [
        [
            [1, 3, args.height, args.width],   # min
            [1, 3, args.height, args.width],   # opt
            [1, 3, args.height, args.width]    # max
        ]
    ]
    print('torch2trt_dynamic')
    model_trt = torch2trt_dynamic(net, [inputs], fp16_mode=True, opt_shape_param=opt_shape_param, input_names=input_names, output_names=output_names)
    save_path = f'retina-{args.network}-{args.height}x{args.width}-b1-fp16.engine'
    print('Saving')
    with open(save_path, 'wb') as f:
        f.write(model_trt.engine.serialize())
