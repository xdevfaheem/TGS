import time
import torch
import fire
from transformers import AutoTokenizer, TextGenerationPipeline, TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM
from transformers import GenerationConfig
from pydantic.errors import NotNoneError
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, root_validator
import json
import uvicorn
from pyngrok import ngrok
import nest_asyncio
from typing import Any, List, Mapping, Optional, Dict
import os, glob
from pathlib import Path
from torch import version as torch_version
import logging
from threading import Thread
import websockets
import asyncio
from huggingface_hub import snapshot_download
from starlette.websockets import WebSocket
from transformers_stream_generator import init_stream_support
from exllama.generator import ExLlamaGenerator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer

def main(model_type="", repo_id="", model_basename="", revision=None, safetensor=True, trust_remote_code=False):

    init_stream_support()

    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Configure file handler for logger
    # Check if a file handler already exists
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler('dummy.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # Add the file handler to the logger
        logger.addHandler(file_handler)

    app = FastAPI(title="LLM", description="Using FastAPI and ngrok for Hosting LLM in colab and getting predictions")

    #global model_type, model, tokenizer, generator, config
    #model_type, repo_id, model_basename, revision, safetensor, trust_remote_code = model_type, repo_id, model_basename, revision, safetensor, trust_remote_code

    class GenerateArgs(BaseModel):
        text: str = Field(..., description="The Prompt That will be Given to the LLM")
        max_tokens: int = Field(256, description="Max Tokens to Generate")
        temperature: float = Field(0.3, description="Temperature for Sampling")
        top_p: float = Field(0.80, description="Top Probabilities for Sampling")
        top_k: int = Field(30, description="Top k numbers of Tokens for Sampling")
        stop: Optional[List[str]] = Field(description="A list of strings to stop generation when encountered.")
        repetition_penalty: float = Field(1.2, description= "Repetition Penalty for most recent tokens")
        token_repetition_penalty_sustain: int = Field(256, description="Most recent tokens to repeat penalty for, -1 to apply to whole context")
        token_repetition_penalty_decay: int = Field(128, description="Gradually decrease penalty over this many tokens")
        typical: float = Field(0.5, description="Locally typical sampling threshold")

    class ModelArgs(BaseModel):
        model_type: str = Field(..., description="Type of the Model. eg., llama, mpt, falcon, rwkv")
        repo_id: str = Field(..., description="Repo id of the Huggingface Model")
        revision: str = Field(..., description="Specific Branch to download the model repo from")
        model_basename: str = Field(..., description="Name of the Safetensor File, use all of that name except '.safetensors'")
        safetensor: bool = Field(..., description="Whether or not to use Safetensor")
        trust_remote_code: bool = Field(..., description="Whether or not to use remote code")

    model_args = ModelArgs(
        model_type=model_type,
        repo_id=repo_id,
        revision=revision,
        model_basename=model_basename,
        safetensor=safetensor,
        trust_remote_code=trust_remote_code
        )



    def vicuna11_prompt(input):
        return f"""USER: {input}\nASSISTANT:"""

    def tulu_prompt(input):
        return f"\n{input}\n\n"

    def landmark_prompt(input):
        return f"USER: {input}\nASSISTANT:"

    def wizardlm_prompt(input):
        return f"{input}\n\n### Response:"

    def vicuna13_prompt(input):
        return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

    USER: {input}
    ASSISTANT:

    """

    def airobros_prompt(input):
        return f"A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. USER: {input} ASSISTANT: "

    def sam_prompt(input):
        return f"""You are Samantha, a sentient AI.

    USER: {input}
    ASSISTANT:"""


    @app.websocket('/ws/completion_stream')
    async def complete_stream(websocket: WebSocket):

        if model is None:
            raise HTTPException(status_code=500, detail="Model has not been loaded")

        await websocket.accept()

        param_dict = dict(GenerateArgs(**await websocket.receive_json()))
        prompt = str(param_dict.get("text"))
        print(prompt)


        if model_type != 'llama':

            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

            gen_config = GenerationConfig(
                max_new_tokens=param_dict.get("max_tokens"),
                temperature=param_dict.get("temperature"),
                top_k=param_dict.get("top_k"),
                top_p=param_dict.get("top_p"),
                do_stream=True
                )

            token_generator = model.generate(
                **tokens,
                generation_config=gen_config,
                do_sample=True,
            )

            async def decode_generator(generator):

                last_tokens = []
                last_decoded_tokens = []
                for x in generator:
                    tokens = last_tokens + x.tolist()
                    word = tokenizer.decode(tokens, skip_special_tokens=True)
                    if "�" in word:
                        last_tokens = tokens
                    else:
                        if " " in tokenizer.decode(last_decoded_tokens + tokens, skip_special_tokens=False):
                            word = " " + word
                        last_tokens = []
                        last_decoded_tokens = tokens

                        yield word

            async for token in decode_generator(token_generator):
                await websocket.send_text(token)

        elif model_type == 'llama':

            ExLlamaGenerator.Settings()
            generator.disallow_tokens(None)
            generator.settings.temperature = param_dict.get("temperature")
            generator.settings.top_p = param_dict.get("top_p")
            generator.settings.top_k = param_dict.get("top_k")
            generator.settings.token_repetition_penalty_max = param_dict.get("repetition_penalty")
            generator.settings.token_repetition_penalty_sustain = param_dict.get("token_repetition_penalty_sustain")
            generator.settings.token_repetition_penalty_decay = param_dict.get("token_repetition_penalty_decay")

            async def exllama_generator(prompt, max_new_tokens, generator=generator):

                new_text = ""
                last_text = ""

                generator.end_beam_search()

                ids = tokenizer.encode(prompt)
                generator.gen_begin_reuse(ids)

                initial_len = generator.sequence[0].shape[0]
                has_leading_space = False

                for i in range(max_new_tokens):
                    token = generator.gen_single_token()

                    if i == 0 and generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
                        has_leading_space = True

                    decoded_text = tokenizer.decode(generator.sequence[0][initial_len:])
                    if has_leading_space:
                        decoded_text = ' ' + decoded_text

                    # Get new token by taking difference from last response:
                    new_token = decoded_text.replace(last_text, "")
                    last_text = decoded_text

                    yield new_token

                    if token.item() == tokenizer.eos_token_id:
                        break

            async for token in exllama_generator(prompt=prompt, max_new_tokens = param_dict.get("max_tokens")):
                await websocket.send_text(token)

    @app.post('/completion_stream')
    def complete_stream(args: GenerateArgs):

        if model is None:
            raise HTTPException(status_code=500, detail="Model has not been loaded")

        param_dict = dict(args)

        prompt = str(param_dict.get("text"))

        print(prompt)


        if model_type != 'llama':

            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

            gen_config = GenerationConfig(
                max_new_tokens=param_dict.get("max_tokens"),
                temperature=param_dict.get("temperature"),
                top_k=param_dict.get("top_k"),
                top_p=param_dict.get("top_p"),
                do_stream=True
                )

            token_generator = model.generate(
                **tokens,
                generation_config=gen_config,
                do_sample=True,
            )

            def decode_generator(generator):

                last_tokens = []
                last_decoded_tokens = []
                for x in generator:
                    tokens = last_tokens + x.tolist()
                    word = tokenizer.decode(tokens, skip_special_tokens=True)
                    if "�" in word:
                        last_tokens = tokens
                    else:
                        if " " in tokenizer.decode(last_decoded_tokens + tokens, skip_special_tokens=False):
                            word = " " + word
                        last_tokens = []
                        last_decoded_tokens = tokens

                        yield word

            return StreamingResponse(decode_generator(token_generator), media_type='text/event-stream')

        elif model_type == 'llama':

            ExLlamaGenerator.Settings()
            generator.disallow_tokens(None)
            generator.settings.temperature = param_dict.get("temperature")
            generator.settings.top_p = param_dict.get("top_p")
            generator.settings.top_k = param_dict.get("top_k")
            generator.settings.token_repetition_penalty_max = param_dict.get("repetition_penalty")
            generator.settings.token_repetition_penalty_sustain = param_dict.get("token_repetition_penalty_sustain")
            generator.settings.token_repetition_penalty_decay = param_dict.get("token_repetition_penalty_decay")

            def exllama_generator(prompt, max_new_tokens, generator=generator):

                new_text = ""
                last_text = ""

                generator.end_beam_search()

                ids = tokenizer.encode(prompt)
                generator.gen_begin_reuse(ids)

                initial_len = generator.sequence[0].shape[0]
                has_leading_space = False

                for i in range(max_new_tokens):
                    token = generator.gen_single_token()

                    if i == 0 and generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
                        has_leading_space = True

                    decoded_text = tokenizer.decode(generator.sequence[0][initial_len:])
                    if has_leading_space:
                        decoded_text = ' ' + decoded_text

                    # Get new token by taking difference from last response:
                    new_token = decoded_text.replace(last_text, "")
                    last_text = decoded_text

                    yield new_token

                    # [End conditions]:
                    #if break_on_newline and # could add `break_on_newline` as a GenerateRequest option?
                    #if token.item() == tokenizer.newline_token_id:
                    if token.item() == tokenizer.eos_token_id:
                        #print(f"eos_token_id: {tokenizer.eos_token_id}")
                        break

            return StreamingResponse(exllama_generator(prompt, max_new_tokens=param_dict.get("max_tokens")), media_type='text/event-stream')

    @app.websocket('/ws/completion')
    async def complete(websocket: WebSocket):

        if model is None:
            raise HTTPException(status_code=500, detail="Model has not been loaded")

        await websocket.accept()

        param_dict = dict(GenerateArgs(**await websocket.receive_json()))

        prompt = str(param_dict.get("text"))
        print(prompt)


        if model_type != 'llama':

            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

            gen_config = GenerationConfig(
                max_new_tokens=param_dict.get("max_tokens"),
                temperature=param_dict.get("temperature"),
                top_k=param_dict.get("top_k"),
                top_p=param_dict.get("top_p"),
                do_stream=False
                )

            output = model.generate(
                **tokens,
                generation_config=gen_config,
                do_sample=True,
            )

            result = tokenizer.decode(output[0], skip_special_tokens=False)

            await websocket.send_text(result)

        elif model_type == 'llama':

            generator.disallow_tokens(None)
            generator.settings.temperature = param_dict.get("temperature")
            generator.settings.top_p = param_dict.get("top_p")
            generator.settings.top_k = param_dict.get("top_k")
            generator.settings.typical = param_dict.get("typical")
            generator.settings.token_repetition_penalty_max = param_dict.get("repetition_penalty")
            generator.settings.token_repetition_penalty_sustain = param_dict.get("token_repetition_penalty_sustain")
            generator.settings.token_repetition_penalty_decay = param_dict.get("token_repetition_penalty_decay")

            output = generator.generate_simple(prompt, max_new_tokens=param_dict.get("max_tokens"))[len(prompt):].lstrip()
            await websocket.send_text(output)

    @app.post('/completion')
    def complete(args: GenerateArgs):

        if model is None:
            raise HTTPException(status_code=500, detail="Model has not been loaded")

        param_dict = dict(args)
        prompt = str(param_dict.get("text"))
        print(prompt)

        if model_type != 'llama':

            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

            gen_config = GenerationConfig(
                max_new_tokens=param_dict.get("max_tokens"),
                temperature=param_dict.get("temperature"),
                top_k=param_dict.get("top_k"),
                top_p=param_dict.get("top_p"),
                do_stream=False
                )

            output = model.generate(
                **tokens,
                generation_config=gen_config,
                do_sample=True,
            )

            result = tokenizer.decode(output[0], skip_special_tokens=False)

            return (result)

        elif model_type == 'llama':

            generator.disallow_tokens(None)
            generator.settings.temperature = param_dict.get("temperature")
            generator.settings.top_p = param_dict.get("top_p")
            generator.settings.top_k = param_dict.get("top_k")
            generator.settings.typical = param_dict.get("typical")
            generator.settings.token_repetition_penalty_max = param_dict.get("repetition_penalty")
            generator.settings.token_repetition_penalty_sustain = param_dict.get("token_repetition_penalty_sustain")
            generator.settings.token_repetition_penalty_decay = param_dict.get("token_repetition_penalty_decay")

            output = generator.generate_simple(prompt, max_new_tokens=param_dict.get("max_tokens"))[len(prompt):].lstrip()

            return (output)


    @app.on_event("startup")
    async def init_model(
        model_type=model_args.model_type,
        repo=model_args.repo_id,
        revision=model_args.revision,
        model_basename=model_args.model_basename,
        safe_tensor=model_args.safetensor,
        trust_remote_code=model_args.trust_remote_code
        ):

        global tokenizer, model, generator, config

        tokenizer = None
        model = None
        generator = None
        config = None


        print("Starting up the LLM API...")
        logger.info("Starting up the LLM API")

        if revision:
            model_repo_path = snapshot_download(repo_id=repo, revision=revision)

        else:
            model_repo_path = snapshot_download(repo_id=repo)

        if model is None:

            if model_type != 'llama':

                print("Initialising a non llama model...")

                try:

                    tokenizer = AutoTokenizer.from_pretrained(
                        model_repo_path,
                        use_fast=False
                    )

                    autogptq_config = AutoConfig.from_pretrained(model_repo_path, trust_remote_code=trust_remote_code)
                    autogptq_config.max_position_embeddings = 4096

                    model = AutoGPTQForCausalLM.from_quantized(
                        config=autogptq_config,
                        model_basename=model_basename,
                        use_triton=False,
                        use_safetensors=safe_tensor,
                        device="cuda:0",
                        quantize_config=None, #max_memory={i: "15GIB" for i in range(torch.cuda.device_count())}
                    )

                    logger.info("Model loaded successfully")

                except Exception as e:

                    logger.exception("Failed to load the model")
                    raise HTTPException(status_code=500, detail="Failed to load the model") from e

            elif model_type == 'llama':

                print("Initialising a llama based model...")

                try:

                    tokenizer_path = os.path.join(model_repo_path, "tokenizer.model")
                    model_config_path = os.path.join(model_repo_path, "config.json")
                    st_pattern = os.path.join(model_repo_path, "*.safetensors")
                    model_path = glob.glob(st_pattern)[0]

                    config = ExLlamaConfig(str(model_config_path))
                    config.model_path = str(model_path)
                    config.max_seq_len = 4096
                    config.compress_pos_emb = 2

                    if torch_version.hip:
                        config.rmsnorm_no_half2 = True
                        config.rope_no_half2 = True
                        config.matmul_no_half2 = True
                        config.silu_no_half2 = True

                    model = ExLlama(config)
                    tokenizer = ExLlamaTokenizer(tokenizer_path)
                    cache = ExLlamaCache(model)
                    generator = ExLlamaGenerator(model, tokenizer, cache)

                except Exception as e:

                    logger.exception("Failed to load the model")
                    raise HTTPException(status_code=500, detail="Failed to load the model") from e
        else:

            logger.info("Model loaded successfully")


    @app.on_event("shutdown")
    async def shutdown_event():
        print("Shutting down the LLM API")
        logger.info("Shutting down the LLM API")

    from flask_cloudflared import _run_cloudflared
    public_url = _run_cloudflared(8000, 8001)
    print("Public UL: ", public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)

if __name__ == '__main__':
  fire.Fire(main)
