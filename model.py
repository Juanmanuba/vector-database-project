import torch
from sentence_transformers import SentenceTransformer


class Model:
    def def_encoding_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer("multi-qa-mpnet-base-dot-v1", device=device)

    def create_vectors_for_index(self, model, data):
        vectors = []

        for ind in data.index:
            text = data["movie_name"][ind] + ": " + data["synopsis"][ind]
            xq = model.encode(text)
            vectors.append(
                (
                    "M" + str(ind),
                    xq.tolist(),
                    {
                        "text": text,
                        "genre": data["genre"][ind],
                    },
                )
            )
        return vectors

    def load_from_csv(self, path):
        pass

    def load_from_mysql(self):
        pass

    def features_def(self, dataset, drop_cols, y):
        pass

    def model_export(self, clf, score):
        pass
