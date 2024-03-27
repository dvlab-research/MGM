import os
import json
from argparse import ArgumentParser

from utils.eval_utils import evaluate
from utils.data_utils import save_json


def main():
    parser = ArgumentParser()
    parser.add_argument('--result_file', type=str, default='llava1.5_13b_val.txt',
                        help='name of saved json')
    parser.add_argument('--output_path', type=str, default='llava1.5_13b_val.json',
                        help='name of saved json')

    args = parser.parse_args()
    out_samples = [json.loads(line) for line in open(args.result_file)]
    out_json = {}
    for _sample in out_samples:
        _result = _sample['parsed_pred']
        if isinstance(_result, list):
            _result = str(_result[0])
        out_json[_sample['id']] = _result
    
    save_json(args.output_path, out_json)


if __name__ == '__main__':
    main()
