import json
import httpx
import requests
import asyncio
import websockets
from websockets.sync.client import connect
from typing import Any, Dict, Optional, List, Mapping, Generator
from pydantic import Extra, Field, root_validator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.base import Generation
from langchain.callbacks import streaming_stdout
callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
from langchain.schema import Generation, LLMResult
from typing import Any
import fire

class Hosted_LLM(LLM):
    """Langchain Wrapper for hosted LLM"""

    endpoint : Optional[str] = ''
    """url of the hosted endpoint tunnel"""

    max_tokens: Optional[int] = 256
    """The maximum number of tokens to generate."""

    temperature: Optional[float] = 0.6
    """The temperature to use for sampling."""

    top_p: Optional[float] = 0.70
    """The top-p value to use for sampling."""

    top_k: Optional[int] = 35
    """The top-k value to use for sampling."""

    streaming: bool = True
    """Whether or not to Stream the Result"""

    ht_ws : Optional[str] = "ws"
    """Whether to send and receive response through http or websockets"""

    completion_url: Optional[str] = ""

    completion_stream_url: Optional[str] = ""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.completion_url: Optional[str] = self.endpoint + 'completion' if self.ht_ws == 'http' else "ws" + self.endpoint[5:] + "ws/completion" if self.ht_ws == 'ws' else None

        self.completion_stream_url: Optional[str] = self.endpoint + 'completion_stream' if self.ht_ws == 'http' else "ws" + self.endpoint[5:] + "ws/completion_stream" if self.ht_ws == 'ws' else None


    @property
    def _llm_type(self) -> str:
        return "Custom Hosted LLM"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "url": self.endpoint,
            "streaming": self.streaming
        }

    def vicuna13_prompt(self, input):
        return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

    USER: {input}
    ASSISTANT:"""

    def ehartford_prompt(self, input):
        return f'''You are a helpful AI assistant.

    USER: {input}
    ASSISTANT:
    '''

    def wizard_prompt(self, input):
        return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input} ASSISTANT:
        """

    def alpaca_prompt(self, instruction: str, input_ctxt: str = None) -> str:
        if input_ctxt:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input_ctxt}

    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:"""


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:

        if stop is not None:
            pass

        llm_args_dict = {
            'text': (prompt),
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'stop': stop
        }

        llm_input_json = json.dumps(llm_args_dict)

        if self.streaming:

            response = ""
            for token in self.stream(prompt=prompt, stop=stop, llm_args=llm_input_json, run_manager=run_manager):
                print(token, end='', flush=True)
            return response

        else:

            if self.ht_ws == 'http':

                with httpx.Client() as client:
                    response = client.post(url=self.completion_url, data=llm_input_json)
                    return response.json()

            elif self.ht_ws == 'ws':

                with connect(self.completion_url) as ws:
                    ws.send(llm_input_json)
                    response = ws.recv()
                    return response
            


    def stream(
        self,
        prompt: str,
        llm_args: dict,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Generator[Dict, None, None]:

        # ws = connect(self.completion_stream_url)
        # ws.send(llm_args)
        # for message in ws:
        #     yield str(message)

            # for message in ws.:
            #     yield message
            # while True:
            #     word = ws.recv()
            #     yield word
            # #for word in websocket.recv():
            #     yield word

        if self.ht_ws == 'http':

                    ## HTTP ##
            with httpx.Client() as client:
                with client.stream("POST", url=self.completion_stream_url, data=llm_args, timeout=10.0) as r:
                    for text in r.iter_text():
                        yield text

        elif self.ht_ws == 'ws':

                    ## WEBSOCKET ##
            with connect(self.completion_stream_url) as ws:
                    ws.send(llm_args)
                    for token in ws:
                        yield token
        
def main(endpoint, streaming=True, max_tokens=512, ht_ws="http", temperature=0.5, top_k=40, top_p=0.95):
    print("Initializing model...")
    model = Hosted_LLM(endpoint=endpoint, streaming=streaming, max_tokens=max_tokens, ht_ws=ht_ws, temperature=temperature, top_p=top_p, top_k=top_k)
    print("Initialized successfully...")
    while True:
        user = input("\nYOU: ")
        if user != 'q':
            response = model(prompt=user)
            print("\nAI:", response)
        else:
            break
if __name__=="__main__":
    fire.Fire(main)
    
