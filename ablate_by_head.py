import os
os.environ['HF_HOME'] = '/home/tatsuya/data/hf_models/'
import torch, argparse, pathlib, tqdm, subprocess
from transformers import AutoConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"cuda available: {torch.cuda.is_available()}")
print(f"number of gpus: {torch.cuda.device_count()}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
print(f"saving and loading HF models from {os.environ['HF_HOME']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', required=False,
                        help="path to the model, if contains multiple checkpoints, runs all checkpoints")
    parser.add_argument('-am', '--ablation_mode', choices=['full', 'pp'], default=None,
                        help="type of ablation to perform: ['full', 'pp'], default=None")
    parser.add_argument('-r', '--revision', required=False,
                        help="checkpoint for Pythia")
    parser.add_argument('-s', '--get_surp', action='store_true',
                        help="whether or not to run get_surprisal.py, default=False")
    parser.add_argument('-l', '--get_loss', action='store_true',
                        help="whether or not to run get_icl_score.py, default=False")

    args = parser.parse_args()
    model_dir, ablation_mode, revision, get_surp, get_loss =\
    args.model_dir, args.ablation_mode, args.revision, args.get_surp, args.get_loss

    # Get the number of layers and attention heads
    config = AutoConfig.from_pretrained(model_dir)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    pbar = tqdm.tqdm(total=num_layers*num_heads)
    for l in range(num_layers):
        for h in range(num_heads):
            ablation_head = f"{str(l)}-{str(h)}"
            if get_surp:
                surp_commands = [
                    "python", "get_surprisal.py",
                    '-m', model_dir,
                    "-ah", ablation_head,
                    '-am', ablation_mode
                ]
                result = subprocess.run(surp_commands, capture_output=True, text=True)
                # Print the output
                print(result.stderr)
                print(result.stdout)

            if get_loss:
                if 'pythia' in model_dir:
                    data_fn = 'pile_10m_tokens_pythia-01.pkl'
                elif 'gpt' in model_dir:
                    data_fn = 'pile_10m_tokens-01.pkl'
                data_fp = os.path.join(DATA_DIR, data_fn)

                loss_commands = [
                    "python", "get_icl_scores.py",
                    "-m",  model_dir,
                    "-d", data_fp,
                    "-am", ablation_mode,
                    "-ah", ablation_head,
                    '-q', '-p',
                    '-p1', '50',
                    '-p2', '500'
                ]
                result = subprocess.run(loss_commands, capture_output=True, text=True)
                print(result.stderr)
                print(result.stdout)
            pbar.update()

if __name__ == "__main__":
    main()