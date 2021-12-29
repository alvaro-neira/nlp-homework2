import csv

import datasets
from datasets.tasks import TextClassification

_DESCRIPTION = """\
A Description
"""

_CITATION = """\
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/alvaro-neira/nlp-homework2/main/train_dataset.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/alvaro-neira/nlp-homework2/main/validation_dataset.csv"


class DataSetTarea2(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="Data set en espanol para tarea 2 NLP",
            features=datasets.Features(
                {
                    "Clase": datasets.Value("string"),
                    "Mensaje": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None

        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=";", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                id, clase, mensaje = row
                yield id_, {"clase": clase, "mensaje": mensaje}
