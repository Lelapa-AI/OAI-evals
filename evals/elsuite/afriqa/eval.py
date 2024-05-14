import ast
import logging
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn

logger = logging.getLogger(__name__)


class Sample(BaseModel):
    question: str
    answers: list[str]

    class Config:
        arbitrary_types_allowed = True


def get_dataset(url: str) -> list[Sample]:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query = {k: v[0] for k, v in query.items()}

    dataset = load_dataset("masakhane/afriqa", **query)
    data = []
    for sample in dataset:
        try:
            corrected_answer = sample["translated_answer"].replace("''", "'").replace("'s", "\'s")
            answers = ast.literal_eval(corrected_answer)
            data.append(Sample(question=sample["translated_question"], answers=answers))
        except SyntaxError as e:
            print(f"Failed to parse answer: {sample['translated_answer']} with error: {e}")
    return data


class AFRIQA(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        dataset: str,
        language: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "MMMU only supports one completion fn"
        self.dataset = dataset
        self.language = language

    def eval_sample(self, sample: Sample, rng):
        assert isinstance(sample, Sample)
        correct_answer = sample.answers
        prompt = "Answer the following question. Provide the answer with the least number of words possible, If you don't know the answer just say you don't know, do not repeat the question.\n\n"
        question = sample.question
        system_prompt = f"You are a helpful assistant able to provide answer in {self.language}."
        try:
            result = self.completion_fn(
                prompt=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt,
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt+question,
                            },
                        ]
                    },
                ],
                temperature=0.0,
                max_tokens=1028,
            )
            sampled = result.get_completions()[0]
        except Exception as e:
            logging.info("Sampling failed!")
            logging.info(sample)
            logging.info(f"Prompt: {question}")
            logging.info(f"Error: {str(e)}")
            sampled = "ERROR: " + str(e)

        return evals.record_and_check_match(
            prompt=question,
            sampled=sampled,
            expected=correct_answer,
        )

    def run(self, recorder):
        samples = get_dataset(self.dataset)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "f1": evals.metrics.get_f1(events),
            "boostrap_std": evals.metrics.get_bootstrap_accuracy_std(events),
        }
