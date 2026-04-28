import os

model_name = "meta-llama/Llama-3.2-1B-Instruct"
users = [1,2,4,8,16,32,64]
input_toks = [1000]
output_toks = [1,3,5,7]
pcmls = [0, 0.5, 0.8, 0.9]
for user in users:
    for input_tok in input_toks:
        for output_tok in output_toks:
            for pcml in pcmls:

                pcml = int(pcml*input_tok)
                url = "http://localhost:8000/v1"
                token = "sk-xxxxxxxxxxxxxx"

                qps = 10
                max_requests = 5
                os.system(
                    f"locust -t 30sec -pcml {pcml} --users {user} -r {user} -o {output_tok} -H {url} -p {input_tok} --api-key {token} --model={model_name} --prompt-randomize --chat --provider openai --temperature 0.0 --fake-image-count 1 --fake-images-per-request 1 --fake-image-resolution 512x512 --header id:f49b2e20-fef3-4441-9358-897f946b8ae2 --summary test-vllm-v5.csv")