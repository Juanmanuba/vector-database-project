import pinecone


def get_index(name):
    pinecone.init(
        api_key="820a6848-41ac-46d8-81eb-9f99e2d1fbee", environment="gcp-starter"
    )

    return pinecone.Index(name, pool_threads=30)
