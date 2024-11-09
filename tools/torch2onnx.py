import sys, os
sys.path.append(os.getcwd())

import os
import argparse
import numpy as np

import torch
from vedacore.misc import Config, load_weights
from vedadet.models import build_detector


def get_names(inp, prefix):
    if not isinstance(inp, (tuple, list)):
        inp = [inp]

    names = []
    for i, sub_inp in enumerate(inp):
        sub_prefix = '{}.{}'.format(prefix, i)
        if isinstance(sub_inp, (list, tuple)):
            names.extend(get_names(sub_inp, sub_prefix))
        else:
            names.append(sub_prefix)

    return names


def flatten(inp):
    if not isinstance(inp, (tuple, list)):
        return [inp]

    out = []
    for sub_inp in inp:
        out.extend(flatten(sub_inp))

    return out


def to(inp, device_or_dtype):
    if not isinstance(inp, (tuple, list)):
        if type(inp).__module__ == torch.__name__:
            if device_or_dtype == 'torch':
                pass
            elif device_or_dtype == 'numpy':
                inp = inp.detach().cpu().numpy()
            else:
                inp = inp.to(device_or_dtype)
        elif type(inp).__module__ == np.__name__:
            if not isinstance(inp, np.ndarray):
                inp = np.array(inp)

            if device_or_dtype == 'torch':
                inp = torch.from_numpy(inp)
            elif device_or_dtype == 'numpy':
                pass
            else:
                inp = inp.astype(device_or_dtype)
        elif isinstance(inp, (int, float)):
            if device_or_dtype == 'torch':
                inp = torch.tensor(inp)
            elif device_or_dtype == 'numpy':
                inp = np.array(inp)
        else:
            raise TypeError(('Unsupported type {}, expect int, float, '
                             'np.ndarray or torch.Tensor').format(type(inp)))

        return inp

    out = []
    for sub_inp in inp:
        out.append(to(sub_inp, device_or_dtype))

    return out


def torch2onnx(
        model,
        dummy_input,
        onnx_model_name,
        dynamic_shape=False,
        opset_version=9,
        do_constant_folding=False,
        verbose=False):

    if isinstance(dummy_input, tuple):
        dummy_input = list(dummy_input)
    dummy_input = to(dummy_input, 'cuda')
    model.eval().cuda()
    with torch.no_grad():
        output = model(dummy_input)

    assert not isinstance(dummy_input, dict), 'input should not be dict.'
    assert not isinstance(output, dict), 'output should not be dict'

    input_names = get_names(dummy_input, 'input')
    output_names = get_names(output, 'output')

    dynamic_axes = dict()
    for name, tensor in zip(input_names+output_names,
                            flatten(dummy_input)+flatten(output)):
        dynamic_axes[name] = list(range(tensor.dim())) if dynamic_shape else [0]

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        verbose=verbose,
        dynamic_axes=dynamic_axes)

    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description='Convert to Onnx model.')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('out', help='output onnx file name')
    parser.add_argument('--dummy_input_shape', default='3,800,1344',
                        type=str, help='model input shape like 3,800,1344. '
                                       'Shape format is CxHxW')
    parser.add_argument('--dynamic_shape', default=False, action='store_true',
                        help='whether to use dynamic shape')
    parser.add_argument('--opset_version', default=18, type=int,
                        help='onnx opset version')
    parser.add_argument('--do_constant_folding', default=False,
                        action='store_true',
                        help='whether to apply constant-folding optimization')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='whether print convert info')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'

    model = build_detector(cfg.model)
    load_weights(model, args.checkpoint)
    model.to(device)
    model.forward = model.forward_impl

    shape = map(int, args.dummy_input_shape.split(','))
    dummy_input = torch.randn(1, *shape)

    if args.dynamic_shape:
        print(f'Convert to Onnx with dynamic input shape and '
              f'opset version {args.opset_version}')
    else:
        print(f'Convert to Onnx with constant input shape '
              f'{args.dummy_input_shape} and '
              f'opset version {args.opset_version}')
    torch2onnx(model, dummy_input, args.out, dynamic_shape=args.dynamic_shape,
               opset_version=args.opset_version,
               do_constant_folding=args.do_constant_folding,
               verbose=args.verbose)
    print(f'Convert successfully, saved onnx file: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
