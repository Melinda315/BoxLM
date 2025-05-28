import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mimic3_0.5")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--beta", default=0.2, type=float)
    parser.add_argument('--layers', type=int, default=1)
    return parser.parse_args()
