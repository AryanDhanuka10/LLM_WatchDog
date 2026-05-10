from infertrack import watchdog

@watchdog(tag="stream", user_id="alice")
def ask_stream(prompt):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},  # ← required for token counts
    )

for chunk in ask_stream("Tell me a story"):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
# CallLog written here with full token counts + cost