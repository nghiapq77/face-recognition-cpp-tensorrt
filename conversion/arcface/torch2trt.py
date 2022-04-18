import argparse
import torch
from torch2trt_dynamic import torch2trt_dynamic
from model_irse import IR_50

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='../../weight/torch/arcface/backbone_ir50_asia.pth', type=str)
parser.add_argument(
    '-o', '--output', default='../../weight/torch/arc/ir50_asia-l2norm-db.onnx', type=str)
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('--width', type=int, default=112)
parser.add_argument('--height', type=int, default=112)
parser.add_argument('-d', '--enable_dynamic_axes',
                    action="store_true", default=False)

args = parser.parse_args()
input_size = [args.height, args.width]
dummy_input = torch.randn(
    [args.batch_size, 3, args.height, args.width], device='cuda')
model = IR_50(input_size)
model.load_state_dict(torch.load(args.model))
model.cuda()
model.eval()
print(model)
# model(dummy_input)
# exit(0)

# export
input_names = ["input"]
output_names = ["output"]

# convert to TensorRT feeding sample data as input
opt_shape_param = [
    [
        [1, 3, args.height, args.width],   # min
        [1, 3, args.height, args.width],   # opt
        [1, 3, args.height, args.width]    # max
    ]
]
print('torch2trt_dynamic')
model_trt = torch2trt_dynamic(model, [dummy_input], fp16_mode=True,
                              opt_shape_param=opt_shape_param, input_names=input_names, output_names=output_names)
save_path = f'arcface-ir50_asia-{args.height}x{args.width}-b1-fp16.engine'
print('Saving')
with open(save_path, 'wb') as f:
    f.write(model_trt.engine.serialize())
