import pandas as pd
from utils import Utils
from index import get_index
from model import Model

# print("Torch version:", torch.__version__)

# print("Is CUDA enabled?", torch.cuda.is_available())


if __name__ == "__main__":
    utils = Utils()
    encoding = Model()

    data = utils.load_from_csv("./data/train.csv")
    print("\nGiven Dataframe :\n\n", data.head(5))

    model = encoding.def_encoding_model()

    print("")
    print("Encoding vectors for index...")
    print("")

    vectors = encoding.create_vectors_for_index(model, data)

    index = get_index("vdp")

    print("")
    print("Beginning upsert of " + str(len(vectors)) + " vectors")
    print("")

    for ids_vectors_chunk in utils.chunks(vectors):
        index.upsert(vectors=ids_vectors_chunk)

    print("Upsert completed")
    print("")

    movie = "Romantic Comedy"
    xq = model.encode(movie).tolist()

    res = index.query(
        xq,
        top_k=3,
        include_metadata=True,
    )

    # Print results
    for result in res["matches"]:
        print(
            f"{round(result['score'], 4)}: {result['metadata']['text']} - genre {result['metadata']['genre']}"
        )
