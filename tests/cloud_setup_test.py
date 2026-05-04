from google.cloud import storage
client = storage.Client()
print(client.project)