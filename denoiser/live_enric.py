import argparse
import torch
import sounddevice as sd
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

import causal_improved_sudormrf_v3 

#(live) python -m denoiser.live_enric --samplerate 16000 --blocksize 16000 -i 3 -o 4 --device cpu
#python -m sounddevice


def load_sudormrf_causal_cpu(model_path, device):
    # 1: declarem el model (instanciem la classe)
    model = causal_improved_sudormrf_v3.CausalSuDORMRF(
        in_audio_channels=1,
        out_channels=512,
        in_channels=256,
        num_blocks=16,
        upsampling_depth=5,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=1,
        )
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.module.to(device)
    model.eval()
    return model

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-o', '--output-device', type=int_or_str,
    help='output device (numeric ID or substring)')
parser.add_argument(
    '-b', '--bypass', action='store_true',
    help='bypass the model application')
parser.add_argument(
    '-c', '--channels', type=int, default=2,
    help='number of channels interface channels')
parser.add_argument('--dtype', help='audio data type')
parser.add_argument('--samplerate', type=float, help='sampling rate')
parser.add_argument('--blocksize', type=int, help='block size')
parser.add_argument('--latency', type=float, help='latency in seconds')
parser.add_argument("--device", default="mps", help='cpu, cuda or mps')
args = parser.parse_args(remaining)




mps_device = torch.device(args.device)
model_path = 'e39_sudo_whamr_16k_enhnoisy_augment.pt'
model = load_sudormrf_causal_cpu(model_path, mps_device)
print("Model loaded.")

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    #shape is [block_size, channels] or [256, 2]
    in_tensor = torch.from_numpy(indata)

    if not args.bypass:
        in_tensor = in_tensor.mean(1) #downmix to mono
        ini_nrg = torch.sum(in_tensor ** 2)
        #in_tensor = (in_tensor - torch.mean(in_tensor)) / torch.std(in_tensor) #OPTIONAL NORMALIZATION 
        in_tensor = model(in_tensor.unsqueeze(0).unsqueeze(0)).detach().squeeze()
        in_tensor /= torch.sqrt(torch.sum(in_tensor ** 2) / ini_nrg) #energy constraint
        in_tensor = in_tensor.unsqueeze(1).repeat(1, 2) # upmix to stereo
        
    outdata[:] = in_tensor#.numpy()
    #print(out_tensor.shape)

try:
    print('starting script')
    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=args.samplerate, blocksize=args.blocksize,
                   dtype=args.dtype, latency=args.latency,
                   channels=args.channels, callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))