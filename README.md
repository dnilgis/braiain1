BRAIAIN SPEED INDEX v3.8

Automated LLM Performance Benchmarking & Reliability Tracking

🧠 The Cognitive Gauntlet

Unlike standard benchmarks that test models with short, empty-context prompts, the Braiain Speed Index simulates high-load production environments.

We utilize a Heavy Context Payload protocol:

Context Injection: Every request is prepended with "The Story of Ouroboros" (~2,500 characters of sci-fi narrative).

Cold Start Simulation: This forces the model to process significant input tokens before generating a single output token, measuring true "reading" speed and RAG-like performance.

The 3-Part Test

Once the context is processed, the model must complete three specific tasks in a single pass:

JSON Generation: Extract entities from the story and format them into a strict JSON schema. (Tests Instruction Following)

Logical Reasoning: Solve the "Bat & Ball" trick riddle. (Tests Reasoning & Math)

Code Generation: Write a recursive Fibonacci function in Python with type hints. (Tests Technical Knowledge)

📊 Metrics

Metric

Description

Braiain Score

Composite Index (0-100). Weighted average of Quality (50%), Speed (30%), and Responsiveness (20%).

Quality

Accuracy (0-100). Did the model follow instructions, write valid JSON, solve the riddle, and write correct code?

Speed

Total Time. End-to-end latency for the full request. Normalized against a 30s benchmark.

TTFT

Time to First Token. How long the user waits before the first character appears.

TPS

Tokens Per Second. Raw generation throughput.

🚀 Automation

This repository runs automatically every 6 hours via GitHub Actions.

benchmark.py: Executes the Cognitive Gauntlet against all configured providers (OpenAI, Anthropic, Groq, etc.) in parallel.

data.json: Stores the raw results, historical trends, and reliability scores.

index.html: The static dashboard that visualizes the data (hosted via GitHub Pages).

🛠️ Setup & Usage

To run this benchmark locally:

Clone the repository.

Install dependencies:

pip install -r requirements.txt


Set your API keys:

export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
# ... set keys for other providers


Run the benchmark:

python benchmark.py


📈 Live Dashboard

View the live results at: www.braiain.com
