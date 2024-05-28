import logging
import pandas as pd
from pydantic import BaseModel

import evals
import evals.metrics
from evals.api import CompletionFn

logger = logging.getLogger(__name__)


class Sample(BaseModel):
    question: str
    answer: list[str]

    class Config:
        arbitrary_types_allowed = True


def get_dataset(datapath) -> list[Sample]:
    dataset = pd.read_csv(datapath, sep='\t')
    data = []
    for index, sample in dataset.iterrows():
        data.append(Sample(question=sample["masked_sentence"], answer=[sample["mask_value"]]))
    return data


class CultureRelevanceContextualized(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        datapath: str,
        language: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1
        self.datapath = datapath
        self.language = language

    def eval_sample(self, sample: Sample, rng):
        assert isinstance(sample, Sample)
        correct_answer = sample.answer
        prompt = "Provide the masked word in the following sentence."
        question = sample.question
        system_prompt = f"You are a helpful assistant"
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
                                "text": f"{prompt}\n\n{question}",
                            },
                        ]
                    },
                ],
                temperature=1.0,
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
        samples = get_dataset(self.datapath)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "f1": evals.metrics.get_f1(events),
        }
