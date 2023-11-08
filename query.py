from index import get_index
from model import Model

encoding = Model()
model = encoding.def_encoding_model()
index = get_index("vdp")

movie = "taken. action movie"

xq = model.encode(movie).tolist()

res = index.query(
    xq,
    top_k=3,
    include_metadata=True,
)

# Print results
for result in res["matches"]:
    print()
    print(
        f"{round(result['score'], 4)}: {result['metadata']['text']} - genre {result['metadata']['genre']}"
    )
