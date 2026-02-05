import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    resp = client.responses.create(
        model="gpt-4.1-mini", input='Return a JSON object with {"ok": true}'
    )

    print(resp.output_text)


if __name__ == "__main__":
    main()
