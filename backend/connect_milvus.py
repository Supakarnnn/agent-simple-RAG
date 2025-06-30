from pymilvus import connections, list_collections, Collection

connections.connect(alias="default", host="localhost", port="19530")
colls = list_collections()

for name in colls:
    col = Collection(name)
    col.load()
    print(f"{name}: {col.num_entities} vectors")