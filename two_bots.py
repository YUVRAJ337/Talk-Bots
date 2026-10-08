#!/usr/bin/env python3
"""Alternate a Gemini bot and a Groq bot in a fixed-length conversation."""

import argparse
import os
import sys

from openai import OpenAI, OpenAIError


GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-20b"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class Bot:
    def __init__(self, name, provider, model, client, persona):
        self.name = name
        self.provider = provider
        self.model = model
        self.client = client
        self.history = [
            {
                "role": "system",
                "content": (
                    persona
                    + " Reply in two short sentences. Respond to the other bot "
                    "and stay on the topic. Do not invent statistics or claim "
                    "that outcomes are guaranteed."
                ),
            }
        ]

    def said(self, text):
        self.history.append({"role": "assistant", "content": text})

    def heard(self, text):
        self.history.append({"role": "user", "content": text})

    def reply(self):
        request = {
            "model": self.model,
            "messages": self.history,
            "max_tokens": 200,
        }

        # GPT-OSS can spend its output budget reasoning before writing a reply.
        if self.provider == "Groq":
            request["reasoning_effort"] = "low"
            request["extra_body"] = {"include_reasoning": False}

        response = self.client.chat.completions.create(**request)
        text = response.choices[0].message.content

        if not text or not text.strip():
            raise RuntimeError(f"{self.name} received an empty reply.")

        return text.strip()


def save_transcript(filename, transcript):
    with open(filename, "w", encoding="utf-8") as file:
        file.write("# Two-bot conversation\n\n")
        for turn, (name, provider, message) in enumerate(transcript):
            file.write(f"**[{turn}] {name} ({provider}):** {message}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Alternate Gemini and Groq bots in a conversation."
    )
    parser.add_argument(
        "--seed",
        default="Should cities invest more in public transport?",
        help="Opening message from Ava",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Number of bot replies after the seed; 10 gives each bot 5 replies",
    )
    parser.add_argument(
        "--save",
        default="conversation.md",
        help="Transcript filename (default: conversation.md)",
    )
    args = parser.parse_args()

    if args.turns < 1:
        parser.error("--turns must be at least 1")

    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not gemini_key:
        sys.exit("GEMINI_API_KEY is missing. Set it in Terminal first.")
    if not groq_key:
        sys.exit("GROQ_API_KEY is missing. Set it in Terminal first.")

    gemini_client = OpenAI(
        api_key=gemini_key,
        base_url=GEMINI_BASE_URL,
        timeout=60.0,
        max_retries=3,
    )
    groq_client = OpenAI(
        api_key=groq_key,
        base_url=GROQ_BASE_URL,
        timeout=60.0,
        max_retries=3,
    )

    ava = Bot(
        name="Ava",
        provider="Gemini",
        model=GEMINI_MODEL,
        client=gemini_client,
        persona=(
            "You support the main idea in the opening message. Give practical "
            "reasons for your position."
        ),
    )
    ben = Bot(
        name="Ben",
        provider="Groq",
        model=GROQ_MODEL,
        client=groq_client,
        persona=(
            "You question the main idea in the opening message. Raise "
            "reasonable risks, costs, or trade-offs."
        ),
    )

    # Ava's seed is the opening message; Ben responds first.
    transcript = [(ava.name, "Seed", args.seed)]
    ava.said(args.seed)
    ben.heard(args.seed)
    save_transcript(args.save, transcript)

    print(f"Gemini model: {GEMINI_MODEL}")
    print(f"Groq model: {GROQ_MODEL}")
    print(f"Replies: {args.turns} (the seed is an additional opening message)\n")
    print(f"[0] Ava (seed): {args.seed}\n")

    speaker, listener = ben, ava

    for turn in range(1, args.turns + 1):
        print(f"{speaker.name} ({speaker.provider}) is thinking...", flush=True)

        try:
            message = speaker.reply()
        except (OpenAIError, RuntimeError) as error:
            save_transcript(args.save, transcript)
            sys.exit(
                f"Stopped on reply {turn}: {error}\n"
                f"Partial transcript saved to {args.save}"
            )

        speaker.said(message)
        listener.heard(message)
        transcript.append((speaker.name, speaker.provider, message))

        print(f"[{turn}] {speaker.name} ({speaker.provider}): {message}\n")
        save_transcript(args.save, transcript)

        speaker, listener = listener, speaker

    print(f"Finished. Transcript saved to {args.save}")


if __name__ == "__main__":
    main()