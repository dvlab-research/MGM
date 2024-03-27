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
    
    judge_dict, metric_dict = evaluate(out_samples)
    metric_dict.update({"num_example": len(out_samples)})
    judge_dict['metric_dict'] = metric_dict
    save_dir = '/'.join(args.output_path.split('/')[:-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_json(args.output_path, judge_dict)

    print(metric_dict)


if __name__ == '__main__':
    main()
