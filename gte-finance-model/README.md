---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:8585
- loss:MultipleNegativesRankingLoss
base_model: Alibaba-NLP/gte-multilingual-base
widget:
- source_sentence: What were the components of the increase in costs related to operating
    channels in 2023?
  sentences:
  - We have audited the accompanying consolidated balance sheets of Berkshire Hathaway
    Inc. and subsidiaries (the “Company”) as of December 31, 2023 and 2022, the related
    consolidated statements of earnings, comprehensive income, changes in shareholders’
    equity, and cash flows, for each of the three years in the period ended December
    31, 2023, and the related notes (collectively referred to as the “financial statements”).
  - '•an increase in costs related to our operating channels of $319.1 million, comprised
    of: –an increase in employee costs of $145.1 million primarily due to increased
    salaries and wages expense, incentive compensation, and benefit costs for retail
    employees, primarily from the growth in our business and increased wage rates;
    –an increase in other operating costs of $67.7 million primarily due to increased
    depreciation costs, technology costs, and repairs and maintenance costs;  –an
    increase in variable costs of $66.8 million primarily due to increased credit
    card fees, distribution costs, and packaging cost...'
  - Timothy A. Massa was elected Senior Vice President of Human Resources and Labor
    Relations in June 2018.
- source_sentence: What are the two primary businesses of Comcast Corporation?
  sentences:
  - "American Water Works Company, Inc. and Subsidiary Companies\nConsolidated Balance\
    \ Sheets\n(In millions, except share and per share data)\nDecember 31, 2022\n\
    December 31, 2021\nCAPITALIZATION AND LIABILITIES\nCapitalization:\n \n \nCommon\
    \ stock ($0.01 par value; 500,000,000 shares authorized; 187,200,539 and 186,880,413\
    \ shares\nissued, respectively)\n$\n2 \n$\n2 \nPaid-in-capital\n6,824 \n6,781\
    \ \nRetained earnings\n1,267 \n925 \nAccumulated other comprehensive loss\n(23)\n\
    (45)\nTreasury stock, at cost (5,342,477 and 5,269,324 shares, respectively)\n\
    (377)\n(365)\nTotal common shareholders' equity\n7,693 \n7,298 \nLong-term debt\n\
    10,926 \n10,341 \nRedeemable preferred stock at redemption value\n3 \n3 \nTotal\
    \ long-term debt\n10,929 \n10,344 \nTotal capitalization\n18,622 \n17,642 \nCurrent\
    \ liabilities:\n \n \nShort-term debt\n1,175 \n584 \nCurrent portion of long-term\
    \ debt\n281 \n57 \nAccounts payable\n254 \n235 \nAccrued liabilities\n706 \n701\
    \ \nAccrued taxes\n49 \n176 \nAccrued interest\n91 \n88 \nLiabilities related\
    \ to assets held for sale\n \n83 \nOther\n255 \n217 \nTotal current liabilities\n\
    2,811 \n2,141"
  - Equifax Inc. is a global data, analytics and technology company. We provide information
    solutions for businesses, governments and consumers, and we provide human resources
    business process automation and outsourcing services for employers.
  - 'Comcast Corporation operates two primary businesses: Connectivity & Platforms
    and Content & Experiences.'
- source_sentence: What total amount was paid in 2023 for asset acquisitions in the
    external research and technologies sector?
  sentences:
  - For investments that were accounted for as asset acquisitions, we paid $3.94 billion
    in 2023 for acquired IPR&D primarily related to acquisitions of DICE, Versanis,
    Emergence Therapeutics AG (Emergence), and Mablink Biosciences SAS (Mablink).
  - The 'Glossary of Terms and Acronyms' is included on pages 315-321.
  - The net interest income for the Consumer Banking segment of Bank of America increased
    by $3.644 billion, from $30.045 billion in 2022 to $33.689 billion in 2023.
- source_sentence: What factors led to the increase in Intelligent Edge earnings from
    operations as a percentage of net revenue?
  sentences:
  - Intelligent Edge earnings from operations as a percentage of net revenue increased
    12.4 percentage points primarily due to decreases in cost of products and services
    as a percentage of net revenue and operating expenses as a percentage of net revenue.
  - The company maintains a leadership position in the exhibition industry through
    forward-thinking initiatives such as unique marketing outreach, seamless digital
    technology, innovative theatre amenities, selective market expansion, and strategic
    theatre closures, in addition to exploring acquisitions to extend the AMC brand.
  - ITEM 3. LEGAL PROCEEDINGS Please see the legal proceedings described in Note 21.
    Commitments and Contingencies included in Item 8 of Part II of this report.
- source_sentence: What is described in Item 8 of a financial reporting document?
  sentences:
  - Item 8 refers to Financial Statements and Supplementary Data in the context of
    financial reporting.
  - "december 31,        | annual maturities ( in millions )\n-------------------\
    \ | ---------------------------------\n2011                | $ 463           \
    \                 \n2012                | 2014                             \n\
    2013                | 2014                             \n2014                |\
    \ 497                              \n2015                | 500               \
    \               \nthereafter          | 3152                             \ntotal\
    \ recourse debt | $ 4612                           "
  - The Intelligent Edge business segment under the Aruba brand includes a portfolio
    of solutions for secure edge-to-cloud connectivity, embracing work from anywhere
    environments, mobility, and IoT device connectivity.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on Alibaba-NLP/gte-multilingual-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) on the csv dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) <!-- at revision 9fdd4ee8bba0e2808a34e0e739576f6740d2b225 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - csv
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: NewModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'What is described in Item 8 of a financial reporting document?',
    'Item 8 refers to Financial Statements and Supplementary Data in the context of financial reporting.',
    'december 31,        | annual maturities ( in millions )\n------------------- | ---------------------------------\n2011                | $ 463                            \n2012                | 2014                             \n2013                | 2014                             \n2014                | 497                              \n2015                | 500                              \nthereafter          | 3152                             \ntotal recourse debt | $ 4612                           ',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### csv

* Dataset: csv
* Size: 8,585 training samples
* Columns: <code>query</code> and <code>corpus</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                            | corpus                                                                               |
  |:--------|:---------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                               |
  | details | <ul><li>min: 4 tokens</li><li>mean: 22.5 tokens</li><li>max: 81 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 103.44 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | query                                                                                                              | corpus                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:-------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What does the NVIDIA computing platform focus on accelerating?</code>                                        | <code>Data Center The NVIDIA computing platform is focused on accelerating the most compute-intensive workloads, such as AI, data analytics, graphics and scientific computing, across hyperscale, cloud, enterprise, public sector, and edge data centers. The platform consists of our energy efficient GPUs, data processing units, or DPUs, interconnects and systems, our CUDA programming model, and a growing body of software libraries, software development kits, or SDKs, application frameworks and services, which are either available as part of the platform or packaged and sold separately.</code> |
  | <code>What was the adjustment for Cadillac dealer strategy in 2023?</code>                                         | <code>The adjustment for the Cadillac dealer strategy in 2023 was 175.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  | <code>What is the standard of proof used in inter partes reviews (IPR) compared to federal district courts?</code> | <code>IPRs are conducted before Administrative Patent Judges in the USPTO using a lower standard of proof than used in federal district court and challenged patents are not accorded the presumption of validity.</code>                                                                                                                                                                                                                                                                                                                                                                                            |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Evaluation Dataset

#### csv

* Dataset: csv
* Size: 905 evaluation samples
* Columns: <code>query</code> and <code>corpus</code>
* Approximate statistics based on the first 905 samples:
  |         | query                                                                             | corpus                                                                             |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 7 tokens</li><li>mean: 22.24 tokens</li><li>max: 83 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 70.35 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | query                                                                                                                          | corpus                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
  |:-------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What are the main ingredients used in the company's products?</code>                                                     | <code>The company uses a variety of ingredients in their products, including high fructose corn syrup, sucrose, aspartame, and other sweeteners, as well as ascorbic acid, citric acid, phosphoric acid, caffeine, and caramel color; they also use orange and other fruit juice concentrates, and water is a main ingredient in substantially all products.</code>                                                                                                                                                                                                                                                              |
  | <code>What are the purposes of borrowings under the 2021 credit facility?</code>                                               | <code>The 2021 credit facility is available for working capital, capital expenditures and other corporate purposes, including acquisitions and share repurchases.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
  | <code>What factors are considered in the revenue disaggregation process according to the guidance on segment reporting?</code> | <code>We have considered (1) information that is regularly reviewed by our Chief Executive Officer, who has been identified as the chief operating decision maker (the 'CODM') as defined by the authoritative guidance on segment reporting, in evaluating financial performance and (2) disclosures presented outside of our financial statements in our earnings releases and used in investor presentations to disaggregate revenues. The principal category we use to disaggregate revenues is the nature of our products and subscriptions and services, as presented in our consolidated statements of operations.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_eval_batch_size`: 16
- `learning_rate`: 2e-05
- `num_train_epochs`: 4
- `warmup_ratio`: 0.1

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4655 | 500  | 0.3183        |
| 0.9311 | 1000 | 0.1923        |
| 1.3966 | 1500 | 0.18          |
| 1.8622 | 2000 | 0.1697        |
| 2.3277 | 2500 | 0.1523        |
| 2.7933 | 3000 | 0.1494        |
| 3.2588 | 3500 | 0.1445        |
| 3.7244 | 4000 | 0.1334        |


### Framework Versions
- Python: 3.12.9
- Sentence Transformers: 4.1.0
- Transformers: 4.52.3
- PyTorch: 2.7.0
- Accelerate: 1.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->