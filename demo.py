import gradio as gr
from llama_cpp import Llama
# demo file on github 
# run this file 
# install this file https://huggingface.co/bartowski/llama-3-neural-chat-v1-8b-GGUF/blob/main/llama-3-neural-chat-v1-8b-Q5_K_M.gguf

SYSTEM_PROMPT="You are a good assistant"
TOP_K = 30
TOP_P = 0.9
REPEAT_PENALTY = 1.1
TEMPERATURE=0.1

model = Llama(
    model_path="llama-3-neural-chat-v1-8b-Q5_K_M.gguf",
    n_ctx=8192,
    n_parts=1,
    verbose=False,
    n_threads=7
)




def predict(message, history, response_format=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for human, assistant in history:
        messages.append({"role": "user", "content": human })
        messages.append({"role": "assistant", "content":assistant})
    
    messages.append({"role": "user", "content": message})
    
    response = ""
    for part in model.create_chat_completion(
            messages,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PENALTY,
            stream=True,
            response_format=response_format
        ):
            delta = part["choices"][0]["delta"]
            if "content" in delta:
                response += delta["content"]
                yield response
    history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    gr.ChatInterface(predict).launch(server_name="0.0.0.0", 
                                     server_port=8000, 
                                     share=False)
