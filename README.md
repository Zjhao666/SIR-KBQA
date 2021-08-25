# StrucATT-KBQA

This is the code for the paper:  xxxxxx

If you find this code useful in your research, please cite

```
xxx
```

## **Setups**

All codes were developed and tested in the following environment.

- Ubuntu 16.04
- Python 3.6.7
- Pytorch 1.2.0

Download the code and data:

```
git clone https://github.com/Zjhao666/StrucATT-KBQA.git
pip install requirements.txt
```

## **Download Pre-processed Data**

We evaluate our methods on [WebQuestionsSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763), [ComplexWebQuestions](https://www.tau-nlp.org/compwebq) and [ComplexQuestions](https://github.com/JunweiBao/MulCQA/tree/ComplexQuestions).

The processed data can be downloaded from [link](???) . Please unzip **(in linux system)** and put the folders under the path *data/*. There are folders for the splits of each dataset, which denote as *splitname_datasetname*. Each folder contains:

- **q.txt**: The file of questions.
- **te.txt**: The file of topic entities (Obtained via NER tool for named entity detection and [Google Knowledge Graph Search API](https://developers.google.com/knowledge-graph)).
- **con.txt**: The file of detected constraints, which usually cannot be detected as topic entities (Obtained via a simple dictionary constructed on training data).
- **a.txt**: The file of answers.
- **g.txt**: The file of ground truth query graphs (No such file is provided for ComplexQuestions).
- **dependency.txt:**  The file of dependency parsing result.
- **constituency.txt**:  The file of constituency parsing result.
- **pytorch_model.bin/pytorch_model_L2_768.bin**: Full Bert and mini Bert.

## **Save Freebase dump in Your Machine**

As we are querying Freebase dynamically, you need install a database engine in your machine. We are using [Virtuoso Open-Source](https://github.com/openlink/virtuoso-opensource). You can follow the instruction to install. Once you installed the database engine, you can download the raw freebase dump from [link](https://developers.google.com/freebase) and setup your database. The dump used in our experiments has been pre-processed by removing non-English triplets (See code in *mycode/FreebaseTool/*). You can use the code to clean up the raw Freebase dump, which could speed up the query time but may not influence the results. After the database is installed, you need to replace the *SPARQLPATH* in *mycode/SPARQL_test.py* file with *your/path/to/database*. To do a quick test, you can run:

```
python mycode/SPARQL_test.py
```

The output is as follows (The output maybe different if you didn't do any pre-processing step to the raw dump.):

```
{'head': {'link': [], 'vars': ['name3']}, 'results': {'distinct': False, 'ordered': True, 'bindings': [{'name3': {'type': 'literal', 'xml:lang': 'en', 'value': 'Shelly Wright'}}, {'name3': {'type': 'literal', 'xml:lang': 'en', 'value': 'Jeffrey Probst'}}, {'name3': {'type': 'literal', 'xml:lang': 'en', 'value': 'Lisa Ann Russell'}}]}}
```

If you fail to save Freebase dump in your machine, you **cannot train** a new model using our code **except WBQ (See following section)**. But you can **still test** our pre-trained models. The caches of tested datasets are stored in *data/datasetname*. Each folder contains:

- **kb_cache.json**: A dictionary of the cache of searched query graphs *{query: answer}*.
- **m2n_cache.json**: A dictionary of the cache of searched mid *{mid: surface name}*.
- **query_cache.json**: A set of the cache of search query *set(query)*.

Notice, to speed up training, the bigger the cache, the better, but too large a cache will burst memory.

## Code Explanation

We use **contiguous-params** lib to speed back propagation.

- **CWQ/WBQ_Runner_v2*.sh:** different ablation experiments running shell.
- **CQ_Runner.sh:**  running v21 (full model) version. Because CQ datasets can't design complete ablation experiments.
- **stanford-parser/preprocess_lcquad.py**: preprocessing for dependency & constituency parsing and save the results.
- **mycode/KBQA_Runner_zjh_classify_symbol_v2*.py**: Main logic of training (**only support Bert verision code to run**).
- **mycode/ModelsRL_full_symbol_v2*.py:**  Main logic of model.
- **mycode/SPARQL_test.py**: provide database connection and SPRAQL query support.

## **Train a New Model**

If you want to train your model, for example CWQ, you can input (please set your gpu_id at *):

```
sudo CUDA_VISIBLE_DEVICES=* python mycode/KBQA_Runner_zjh_classify_symbol_v21.py  \
        --dataset CWQ \
        --train_folder  data/train_%s \
        --dev_folder data/dev_%s \
        --test_folder data/test_%s \
        --vocab_file data/%s/vocab.txt \
        --KB_file data/%s/kb_cache.json \
        --M2N_file data/%s/m2n_cache.json \
        --QUERY_file data/%s/query_cache.json \
        --output_dir trained_model/%s_%s \
        --config config/bert_config.json \
        --gpu_id 0\
        --load_model trained_model/%s_%s/my_model_3hop \
        --save_model my_model_3hop \
        --max_hop_num 2 \
        --num_train_epochs 30 \
        --do_train 1\
        --do_eval 1\
        --do_policy_gradient 1\
        --learning_rate 1e-5 \
        --alpha 0.1 \
```

or just run 

```
bash CWQ_Runner_v21.sh
```

Our code version (KBQA_Runner_zjh_classify_symbol_v2*.py) corresponds to our ablation experiment one by one.

| Suffix | Ablation Experiment        |
| ------ | -------------------------- |
| v21    | Full StrucATT / Full model |
| v22    | w/o answer types feature   |
| v23    | w/o constituency analysis  |
| v24    | w/o dependency analysis    |
| v25    | w/o first word emb         |



