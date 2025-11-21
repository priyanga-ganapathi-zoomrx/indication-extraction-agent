from litellm import completion

resp = completion(
    model="gemini/gemini-3-pro-preview",
    messages=[{"role": "user", "content": "Solve a complex math problem step by step."}],
    reasoning_effort="high",
    api_key="sk-cMw4owgKaOteHD4CoWSudA",
    base_url="https://dev-cg-litellm.zoomrx.ai",
)

print(resp)
print("-" * 50)
print(resp.choices[0].message.content)
