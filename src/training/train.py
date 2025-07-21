import os, argparse, torch, glob, json, re, math
from src.training.lm_trainer import LMTrainer
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
from torch import cuda
# os.environ['HF_HOME'] = os.path.join(pathlib.Path(__file__).parent.resolve(), 'models')
# os.environ['CURL_CA_BUNDLE'] = ''  # if SSL Error
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use GPU 0 and GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(
        raw_data, tokenized_data, val_fp, reg_fp, output_dir, config_dict,
        model_name, reg_lambda, device, report_to, noise, smooth):

    for cat in config_dict:
        for key in config_dict[cat]:
            val = config_dict[cat][key]
            if type(val) == str and re.match(r"\d+\.?\d+e-?\d+", val) is not None:
                config_dict[cat][key] = float(val)
            elif type(val) in [list, float] or not val.replace('_', '').isdigit():
                config_dict[cat][key] = val
            else:
                config_dict[cat][key] = int(val)

    if raw_data:
        from_hub = True
        data_fp = raw_data
    elif tokenized_data:
        from_hub = False
        data_fp = tokenized_data
    elif segmented_data:
        from_hub = False
        data_fp = segmented_data

    print('\n' + '=' * 100 + f'Training GPT-2 on {data_fp}) from scratch...')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    trainer = LMTrainer(output_dir=output_dir,
                        model_name=model_name,
                        data_fp=data_fp,
                        val_fp=val_fp,
                        reg_fp=reg_fp,
                        config_dict=config_dict,
                        from_hub=from_hub,
                        reg_lambda=reg_lambda,
                        device=device,
                        report_to=report_to,
                        noise=noise,
                        smooth=smooth,
                        **config_dict['lmtrainer']
                        )
    trainer.train_lm()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to train mini gpt2 model")
    parser.add_argument("--raw_data", default=None, help="name of training corpus")
    parser.add_argument("--tokenized_data", default=None, help="pretokenized .pkl data for training")
    parser.add_argument("--segmented_data", default=None, help="pretokenized and segmented .jsonl data for training")
    parser.add_argument("--validation_data", default=os.path.join('data', 'pile_10m_tokens-01.pkl'),
                        help="pretokenized .pkl data for validation")
    parser.add_argument("--regularization_data", default=None,
                        help="pretokenized .jsonl data for regularization")
    parser.add_argument("--use_wandb", action="store_true", help="should I use wandb for logging?")
    parser.add_argument("--gpu", type=str, default='0', help="which GPU to use; 0 or 1; choose wisely based on GPU usage!")
    parser.add_argument("--config_fp", default=None, help="fp to the .json config file for tokenizer and lm training")
    parser.add_argument("--n_layer", default=0, help="number of layers")
    parser.add_argument("--t_block", default='mlp', help="type of transformer block")
    parser.add_argument("--reg_lambda", type=float, default=0, help="lambda for {syntactic|copying} regularizer")
    parser.add_argument("--smooth", action="store_true", help="regularize every step, default=False")
    parser.add_argument("--noise", type=float, default=0, help="sd for Gaussian noise injected into FFN")
    parser.add_argument("--seed", type=int, required=True, help="seed, default is not set")

    args = parser.parse_args()
    raw_data, tokenized_data, segmented_data, validation_data,\
    regularization_data, use_wandb, gpu, config_fp, n_layer,\
    t_block, reg_lambda, smooth, noise, seed =\
        args.raw_data, args.tokenized_data, args.segmented_data, args.validation_data,\
        args.regularization_data, args.use_wandb, args.gpu, args.config_fp, args.n_layer,\
        args.t_block, args.reg_lambda, args.smooth, args.noise, args.seed

    if not (raw_data or tokenized_data or segmented_data):
        raise IOError("Provide either raw or tokenized data for training.")

    if use_wandb:
        import wandb
        wandb.init(project="gpt2-spicl", entity="t-aoyam")
        report_to = 'wandb'
    else:
        print('not using wandb')
        os.environ["WANDB_DISABLED"] = "true"
        report_to = None
    if gpu:  # if only 1 GPU is selected, mask others
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        device = 'cuda:0' if cuda.is_available() else 'cpu'  # always 0 after hiding other GPUs
    else:
        device = 'cuda' if cuda.is_available() else 'cpu'
    # device = 'cuda'

    with open(config_fp) as f:
        config_dict = json.load(f)
        config_dict['lm']['n_layer'] = n_layer

    batch_size = int(config_dict['lm_training']['per_device_train_batch_size'])*\
                 int(config_dict['lm_training']['gradient_accumulation_steps'])
    if seed:
        config_dict["lmtrainer"]["seed"] = str(seed)
    reg_lambda_code = str(int(math.log10(1/reg_lambda))) if reg_lambda else '0'
    # e.g. gpt2-attn-l2-b4-r3 -> gpt2 with only attention, 2 layers, batch size of 4, lambda = 1/(10^3)
    reg_lambda_code = 'r' + reg_lambda_code  # r for regularization
    if regularization_data:
        reg_lambda_code = 'i'+reg_lambda_code  # i for induction
    if smooth:
        reg_lambda_code = 'c'+reg_lambda_code  # c for continuous
    if noise:
        reg_lambda_code = 'n' + str(noise)[0] + str(noise)[2:]
    model_name = f'gpt2-{t_block}-l{n_layer}-b{str(batch_size)}-{reg_lambda_code}'
    model_name += f'-s{str(config_dict["lmtrainer"]["seed"])}'
    output_dir = os.path.join("models", model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main(
        raw_data=raw_data,
        tokenized_data=tokenized_data,
        val_fp=validation_data,
        reg_fp=regularization_data,
        output_dir=output_dir,
        config_dict=config_dict,
        model_name=model_name,
        reg_lambda=reg_lambda,
        device=device,
        report_to=report_to,
        noise=noise,
        smooth=smooth
    )