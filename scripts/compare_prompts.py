import json

from walmart_genai.core.generate import answer_question

QUESTIONS = [
    "What is the return policy for electronics?",
    "Can I return an opened laptop after 45 days?",
    "Do you store my credit card number if I use Walmart Pay?",
]


def main():
    for q in QUESTIONS:
        print("\n" + "=" * 80)
        print("Q:", q)

        for mode in ["base", "structured"]:
            out = answer_question(q, mode=mode)
            print(f"\n--- mode={mode} ---")
            print(json.dumps(out.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
