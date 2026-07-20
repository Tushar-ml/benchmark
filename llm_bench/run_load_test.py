import os

model_name = "Qwen/Qwen3-0.6B"
users = [1,2,4,8,16,32,64]
input_toks = [1000]
output_toks = [500]
pcmls = [0.0]
for user in users:
    for input_tok in input_toks:
        for output_tok in output_toks:
            for pcml in pcmls:

                pcml = int(pcml*input_tok)
                url = "https://http.******.clusters.simplismart.tech"
                token = "<jwt-token>"

                qps = 10
                max_requests = 5
                os.system(
                    f"locust -t 30s -pcml {pcml} --users {user} -r {user} -o {output_tok} -H {url} -p {input_tok} --api-key {token} --model={model_name} --prompt-randomize --chat --provider openai --stream --temperature 0.0  --header id:f49b2e20-fef3-4441-9358-897f946b8ae2 --summary test-vllm-v6-proxy.csv")
