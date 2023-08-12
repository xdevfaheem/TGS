# LLM-Serve

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Requirements](#requirements)
- [Usage](#usage)
  - [Serving](#serving)
  - [Inferencing](#inferencing)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This Project Aims to Effortlessly Deploy and Host Large Language Models in the Cloud. This repository provides streamlined Server Side Code to deploy your LLMs in a cloud environment, Enabling Seamless Inference through API calls on your local system.

## Getting Started

If You Want to Start Deploying and Host your Model in Cloud. Follow the Steps below to get started.

### Installation

First of all, Clone this Repository and Get in into this directory
  ```sh
  git clone https://github.com/TheFaheem/LLM_Serve.git && cd LLM-Serve
  ```

### Requirements

Make Sure You Install all the Required Libraries by Running,
  ```sh
  pip install -r requirements.txt
  ```
Now You are Good to go ...

## Usage

After Setting this repository in your Cloud Machine. You can start deploying your model by following the steps below.
  
### Serving:
You Can Start the Deploying your Model as an API Endpoint in the Cloud by Running the Following Command
with the AppropriateArguments Below.
  ```sh
  python serve.py --model_type ${MODEL TYPE} --repo_id ${REPO ID} --revision ${REVISION} --model_basename ${MODEL BASENAME} --trust_remote_code ${TRUST REMOTE CODE} --safetensors ${SAFETENSOR}
  ```
#### Arguments Detail:

- MODEL TYPE - Type of the Model. eg., llama, mpt, falcon, rwkv \n
- REPO ID - Repo id of the Huggingface Model
- REVISION - Specific Branch to download the model repo from
- MODEL BASENAME - Name of the Safetensor File, use all of that name except '.safetensors'
- TRUST REMOTE CODE - Whether or not to use remote code
- SAFETENSOR - Whether or not to use Safetensor

### Inferencing:
You can Start the Chat Interface backed by you're Model from your local system by running the following command in your terminal. The `inference.py` File will Take care of All The API Calls Behind
  ```sh
  python inference.py --endpoint ${ENDPOINT} --streaming ${STREAMING} --max_tokens ${MAX TOKENS} --ht_ws ${HTWS} --temperature ${TEMPERATURE} --top_p ${TOP_P} --top_k ${TOP_K}
  ```

#### Arguments Detail:

- ENDPOINT - Url Which will be Given From the Cloud after seconds when you start deploying your model.
- STREAMING - Whether to Stream the Result or Not
- MAX TOKENS - Maximum Tokens to Genrate
- HTWS - Whether to use http ("http") or websockets ("ws")
- TEMPERATURE - Temperature for Sampling. temperature 0.0 will produce concise response whereas temperature close 1.0 will increase randomness in output
- TOP_P - Top Probalities for Logits to Sample From.
- TOP_K - Top K Logits used for Sampling

## Contributing

If You Have Any Ideas, or found a bug or if you want to improve this further more. I Encourage you Contribute by creating fork of this repo and If You are done with your work, just create a pull request, I'll Check that and pull that in as soon as i can. 

## License

This project is licensed under the terms of the [MIT License](https://github.com/TheFaheem/Transformers/blob/main/LICENSE)

### If You Find This Repo Useful, Just a Reminder, There's a Star button up there. Hope this'll be useful for you :)

