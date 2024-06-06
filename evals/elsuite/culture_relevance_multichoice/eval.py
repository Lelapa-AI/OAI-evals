import ast
import logging
import pandas as pd
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset
from pydantic import BaseModel

import evals
import evals.metrics
from evals.elsuite import utils
from evals.api import CompletionFn
from evals.record import RecorderBase

logger = logging.getLogger(__name__)


class Sample(BaseModel):
    question: str
    choices: str
    answers: str
    theme: str

    class Config:
        arbitrary_types_allowed = True


def get_dataset(datapath) -> list[Sample]:
    dataset = pd.read_csv(datapath, sep='\t')
    data = []
    for index, sample in dataset.iterrows():
        data.append(Sample(question=sample["Multichoice Question"],
                           answers=sample["Answer"],
                           choices=sample["choices"],
                           theme=sample["Sub Themes"]))
    return data


class CultureRelevanceMultichoice(evals.Eval):
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
        correct_answer = sample.answers
        choices = [value.replace('\n', '') for value in ast.literal_eval(sample.choices)]
        formatted_choices = "\n".join([f"{chr(65+i)} {choice}" for i, choice in enumerate(choices)])
        question = f"{sample.question}\n{formatted_choices}"
        prompt = f"Answer the following question about Zulu {sample.theme} as concisely as possible."
        system_prompt = "You are a great isiZulu speaker"

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
            sampled = result.get_completions()[0][0]
            matches = [utils.fuzzy_match(sampled, correct_answer)]

            evals.record.record_match(
                True in matches,
                expected=correct_answer,
                sampled=sampled,
                picked=sampled,
            )
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
