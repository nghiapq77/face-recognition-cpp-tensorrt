import argparse
import torch
from model_irse import IR_50

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='../../weights/torch/arc/backbone_ir50_asia.pth', type=str)
parser.add_argument(
    '-o', '--output', default='../../weights/torch/arc/ir50_asia-l2norm-db.onnx', type=str)
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

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = ["input"] + ["learned_%d" % i for i in range(24)]
output_names = ["output"]

if args.enable_dynamic_axes:
    # Dynamic batch size
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    torch.onnx.export(model, dummy_input, args.output, dynamic_axes=dynamic_axes,
                      verbose=True, input_names=input_names, output_names=output_names)
else:
    # Fixed batch size
    torch.onnx.export(model, dummy_input, args.output,
                      verbose=True, input_names=input_names, output_names=output_names)
