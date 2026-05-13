import asyncio
from benchmark.pipelines.generation import GenerationPipeline

TOPICS = ["AI chips", "large language models", "AI policy", "open source AI"]

async def main() -> None:
    pipe = GenerationPipeline()
    for topic in TOPICS:
        count = await pipe.run(topic, limit=10)
        print(f"{topic}: saved {count}")

if __name__ == "__main__":
    asyncio.run(main())
