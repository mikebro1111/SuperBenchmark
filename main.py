import os
import json
from datetime import datetime
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


class BenchmarkingResult(BaseModel):
    request_id: str
    prompt_text: str
    generated_text: str
    token_count: int
    time_to_first_token: int  # in milliseconds
    time_per_output_token: int  # in milliseconds
    total_generation_time: int  # in milliseconds
    timestamp: datetime


# Global variables
benchmarking_results: List[BenchmarkingResult] = []
SUPERBENCHMARK_DEBUG = os.getenv('SUPERBENCHMARK_DEBUG', 'False').lower() == 'true'


def load_test_data():
    try:
        with open('test_database.json', 'r') as file:
            data = json.load(file)
            for item in data.get('benchmarking_results', []):
                item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                benchmarking_results.append(BenchmarkingResult(**item))
    except Exception as e:
        raise Exception(f"Error loading test database: {e}")


if SUPERBENCHMARK_DEBUG:
    load_test_data()
else:
    @app.middleware("http")
    async def not_ready_middleware(request, call_next):
        raise HTTPException(status_code=503, detail="Feature not ready for live yet. Enable DEBUG mode.")


@app.get("/results/average")
def get_average_results():
    if not benchmarking_results:
        raise HTTPException(status_code=404, detail="No benchmarking results found.")

    return compute_average(benchmarking_results)


@app.get("/results/average/{start_time}/{end_time}")
def get_average_results_in_window(start_time: str, end_time: str):
    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO 8601 format.")

    filtered_results = [
        result for result in benchmarking_results if start_dt <= result.timestamp <= end_dt
    ]

    if not filtered_results:
        raise HTTPException(status_code=404, detail="No benchmarking results found in the specified time window.")

    return compute_average(filtered_results)


def compute_average(results: List[BenchmarkingResult]) -> Dict[str, float]:
    count = len(results)
    total_token_count = sum(r.token_count for r in results)
    total_time_to_first_token = sum(r.time_to_first_token for r in results)
    total_time_per_output_token = sum(r.time_per_output_token for r in results)
    total_generation_time = sum(r.total_generation_time for r in results)

    return {
        "average_token_count": total_token_count / count,
        "average_time_to_first_token": total_time_to_first_token / count,
        "average_time_per_output_token": total_time_per_output_token / count,
        "average_total_generation_time": total_generation_time / count,
    }
