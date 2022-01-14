import socket

def encodedForSending(stringToEncode):
    return stringToEncode.encode('utf-8')

message = "I am your CLIENT"

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("0.0.0.0", 8080))


client.send(encodedForSending(message))

from_server = client.recv(4096)
client.close()
print(from_server.decode('utf-8'))