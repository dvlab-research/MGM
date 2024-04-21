# Evaluation Guidelines
We provide detailed instructions for evaluation. 
To execute our evaluation script, please ensure that the structure of your model outputs is the same as ours.

We provide two options:
1. Evaluation only: you can parse the response on your own and simply provide one file with all the final predictions.
2. Parse and evaluation: you can leave all the responses to us with the output formats shown below.

## Evaluation Only
If you want to use your own parsing logic and *only provide the final answer*, you can use `main_eval_only.py`.

You can provide all the outputs in *one file* in the following format:

```
{
    "validation_Accounting_1": "D", # strictly "A", "B", "C", "D" for multi-choice question
    "validation_Architecture_and_Engineering_14": "0.0", # any string response for open question.
    ...
}
```
Then run eval_only with:
```
python main_eval_only.py --output_path ./example_outputs/llava1.5_13b/total_val_output.json
```

Please refer to [example output](https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/example_outputs/llava1.5_13b/total_val_output.json) for a detailed prediction file form.


## Parse and Evaluation
You can also provide response and run the `main_parse_and_eval.py` to use our answer parsing processing and evaluation pipeline as follows:

### Output folder structure

```
└── model_name
    ├── category_name (e.g., Accounting)
    │   ├── output.json
    └── category_name (e.g., Electronics)
        ├── output.json
    ...
```

### Output file
Each `output.json`` has a list of dict containing instances for evaluation ().
```
[
    {
        "id": "validation_Electronics_28",
        "question_type": "multiple-choice",
        "answer": "A", # given answer
        "all_choices": [ # create using `get_multi_choice_info` in 
            "A",
            "B",
            "C",
            "D"
        ],
        "index2ans": { # create using `get_multi_choice_info` in 
            "A": "75 + 13.3 cos(250t - 57.7°)V",
            "B": "75 + 23.3 cos(250t - 57.7°)V",
            "C": "45 + 3.3 cos(250t - 57.7°)V",
            "D": "95 + 13.3 cos(250t - 57.7°)V"
        },
        "response": "B" # model response
    },
    {
        "id": "validation_Electronics_29",
        "question_type": "short-answer",
        "answer": "30", # given answer
        "response": "36 watts" # model response
    },
    ...
]
```

### Evaluation
```
python main_parse_and_eval.py --path ./example_outputs/llava1.5_13b --subject ALL # all subject

# OR you can sepecify one subject for the evaluation

python main_parse_and_eval.py --path ./example_outputs/llava1.5_13b --subject elec # short name for Electronics. use --help for all short names

```

`main_parse_and_eval.py` will generate `parsed_output.json` and `result.json` in the subfolder under the same category with output.json, respectively.

```
├── Accounting
│   ├── output.json
│   ├── parsed_output.json
│   └── result.json
└── Electronics
    ├── output.json
    ├── parsed_output.json
    └── result.json
...
```

### Print Results
You can print results locally if you want. (use `pip install tabulate` if you haven't)
```
python print_results.py --path ./example_outputs/llava1.5_13b
# Results may be slightly different due to the ramdon selection for fail response
```



##### Run Llava
In case if you want to reproduce the results of some of the models, please go check run_llava.py as an example.

By seeting up the env following the [llava official repo](https://github.com/haotian-liu/LLaVA) and installing `datasets` packages by huggingface, you can run llava viathe following command:

```
CUDA_VISIBLE_DEVICES=0 nohup python run_llava.py \
--output_path example_outputs/llava1.5_13b_val.json \
--model_path liuhaotian/llava-v1.5-13b \
--config_path configs/llava1.5.yaml
```

Then you can evaluate the results via the very first pipeline.
