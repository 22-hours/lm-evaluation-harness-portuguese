# TODO: colocar description_dict ou mudar o few_show_context

# O QUE EU QUERIA:
# ASSIN2_RTE_SINGLE_EXAMPLE_PROMPT ="""\
# ###
# Exemplo {n}:
# Texto: {text}
# Hipótese: {hypothesis}
# Predição: {entailment}"""

# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import numpy as np
from lm_eval.api.metrics import mean
from lm_eval import utils
from collections import defaultdict
import math

from lm_eval.api.task import ConfigurableTask, TaskConfig
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task


# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""

def items_to_trues_and_preds(items):
    trues, preds = zip(*items)
    assert len(trues) == len(preds)
    trues, preds = np.array(trues), np.array(preds)
    return trues, preds


def compute_macro_f1(items):
    trues, preds = items_to_trues_and_preds(items)
    macro_f1 = f1_score(y_true=trues,
                        y_pred=preds,
                        average="macro",
                        labels=list(set(trues)))
    return macro_f1

def compute_person(items):
    trues, preds = items_to_trues_and_preds(items)
    person = pearsonr(trues, preds)[0]
    return person

# def compute_mse(items):
#     trues, preds = items_to_trues_and_preds(items)
#     mse = ((trues - preds) ** 2).mean()
#     return mse


def string_to_float(x):
    """Converts a string to a float, returning -1 if the conversion fails."""
    try:
        return float(x)
    except:
        return -1

class Assin2Base(ConfigurableTask):
    """Base class for ASSIN2 tasks.

    Modifications:
    - `few_show_examples` method allow to select the same number of examples
    for each label. This avoids bias towards the majority class.
    - `fewshot_context` method modified to also pass index of the each few shot
    example (to be used in the prompt) and an indicator if the doc is a few shot
    or a text document.
    
    """
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "assin2"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(map(self._process_doc,
                                               self.dataset["train"]))
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            #return list(map(self._process_doc, self.dataset["validation"]))
            return self.dataset["validation"].map(self._process_doc)

    def test_docs(self):
        if self.has_test_docs():
            #return list(map(self._process_doc, self.dataset["test"]))
            return self.dataset["test"].map(self._process_doc)


@register_task("assin2rte")
class Assin2RTE(Assin2Base):
    VERSION = "balanced_log_likelihood"
    CLASS_NAMES = ["Não", "Sim"]
    CONFIG = TaskConfig(**{
        'description': 'Analise se a hipótese é uma consequência lógica da premissa apresentada.\n\n',
        'num_fewshot': 18,
        'training_split': 'train',
        'validation_split': 'validation',
        'test_split': 'test',
        'target_delimiter': ''
    })

    def _process_doc(self, doc):
        # TODO: Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.
        # NOTE: DELETE THIS FUNCTION IF UNUSED.
        # CLASS_LABELS = ["None", "Entailment"]
        doc["label_name"] = self.CLASS_NAMES[doc["entailment_judgment"]]
        doc["__fewshot_balance_key__"] = doc["label_name"]
        return doc

    def doc_to_text(self, doc, n_example_fewshot=None):
        # TODO: Format the query prompt portion of the document example.
        # return "\n".join([
        #     "###",
        #     # f"Premissa: {doc['premise']}",
        #     # f"Hipótese: {doc['hypothesis']}",
        #     # f"Predição:"
        #     f"[A]: {doc['premise']}",
        #     f"[B]: {doc['hypothesis']}",
        #     f"Resposta:"
        # ])
    
        # Prompt Ramon
        # return "\n".join([
        #     "###",
        #     f"Sabendo que '{doc['premise']}' é verdadeiro, podemos dizer que '{doc['hypothesis']}' é verdadeiro? Sim ou não",
        #     f"Resposta:"
        # ])
        ex_index = f"Exemplo {n_example_fewshot}:" if n_example_fewshot else ""
        return "\n".join([
            f"###\n{ex_index}",
            f"Premissa: {doc['premise']}",
            f"Hipótese: {doc['hypothesis']}",
            f"Acarretamento:"
            # f"[A]: {doc['premise']}",
            # f"[B]: {doc['hypothesis']}",
            # f"Resposta:"
        ])
        # Não fica bom
        return "\n".join([
            f"###\n{ex_index}",
            f"Sabendo que '{doc['premise']}', podemos dizer que '{doc['hypothesis']}'? Sim ou não?",
            f"Resposta:"
        ])

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        target = doc["label_name"]
        return " " + target

    def construct_requests(self, doc, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        #ll_no, _ = rf.loglikelihood(ctx, " Não")
        ll_no = Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, " " + "Não"),
                idx=0,
                **kwargs
        )

        #ll_yes, _ = rf.loglikelihood(ctx, " Sim")
        ll_yes = Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, " " + "Sim"),
                idx=1,
                **kwargs
        )

        return [ll_no, ll_yes]

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.
        true = doc["entailment_judgment"]
        ll_no, ll_yes = results
        pred = ll_yes[0] > ll_no[0]

        return {"accuracy": (true == pred), "macro_f1": (true, pred)}

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and an aggregation function as value which
        # determines how to combine results from each document in the dataset.
        # Check `lm_eval.metrics` to find built-in aggregation functions.
        return {"accuracy": mean, "macro_f1": compute_macro_f1}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {"macro_f1": True, "accuracy": True}


@register_task("assin2sts")
class Assin2STS(Assin2Base):
    VERSION = "balanced_log_likelihood_remove_low_confidence"
    # class names used only for training
    CLASS_NAMES = ["1", "2", "3", "4", "5"]
    CONFIG = TaskConfig(**{
        'description': '''Avalie o par de sentenças como 1, 2, 3, 4, ou 5.
1. sentenças completamente diferentes, em diferentes tópicos.
2. sentenças não relacionadas, mas no mesmo tópico.
3. sentenças em algum sentido relacionadas: descrevem diferentes fatos, mas compartilham certos detalhes.
4. sentenças proximamente relacionadas, mas diferem em alguns detalhes.
5. sentenças que significam essencialmente a mesma coisa.\n\n''',
        'num_fewshot': 15,
        'training_split': 'train',
        'validation_split': 'validation',
        'test_split': 'test',
        'target_delimiter': ''
    })

    def _remove_low_confidence(self, row):
        """Remove low confidence examples from training set."""
        # keep only integer similarity
        keep = row["relatedness_score"].is_integer()
        if not keep:
            return keep
        # remove examples with high score but no entailment
        if row["relatedness_score"] >= 4 and row["entailment_judgment"] == 1:
            return True
        # remove examples with low score but entailment
        if row["relatedness_score"] <= 2 and row["entailment_judgment"] == 0:
            return True
        # keep examples with score 3
        return row["relatedness_score"] == 3

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                # this will be called just once
                train_ds_filter = self.dataset["train"].filter(
                    self._remove_low_confidence)
                train_ds_filter = train_ds_filter.map(self._process_doc)
                self._training_docs = list(train_ds_filter)
            return self._training_docs

    def doc_to_text(self, doc, n_example_fewshot=None):
        ex_index = f"Exemplo {n_example_fewshot}:" if n_example_fewshot else ""
        return "\n".join([
            f"###\n{ex_index}",
            f"Sentença 1: {doc['premise']}",
            f"Sentença 2: {doc['hypothesis']}",
            f"Similaridade:"
        ])
    
    def _process_doc(self, doc):
        # will be called only for training docs
        doc["relatedness_score_integer"] = str(int(doc["relatedness_score"]))
        doc["__fewshot_balance_key__"] = doc["relatedness_score_integer"]
        return doc


    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        # this is only used for few show documents (train)
        target = doc["relatedness_score_integer"]
        return " " + target

    def construct_requests(self, doc, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # lls = [
        #     rf.loglikelihood(ctx, " {}".format(choice))[0]
        #     for choice in self.CLASS_NAMES
        # ]

        lls = [
                Instance(
                    request_type="loglikelihood",
                    doc=doc,
                    arguments=(ctx, " {}".format(choice)),
                    idx=i,
                    **kwargs,
                )
                for i, choice in enumerate(self.CLASS_NAMES)
            ]
        return lls

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.
        # true is a float
        true = doc["relatedness_score"]
        # pred is a string converted to float
        pred = string_to_float(np.argmax([score for _, score in results])+1)
        return {"pearson": (true, pred), "mse": (true - pred) ** 2}

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and an aggregation function as value which
        # determines how to combine results from each document in the dataset.
        # Check `lm_eval.metrics` to find built-in aggregation functions.
        return {"pearson": compute_person, "mse": mean}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {"pearson": True, "mse": False}
