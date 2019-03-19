import json
import random
import socket
import ssl


rowsize = 10
colsize = 8
weights = [[random.random() * 3 - 1.5 for i in range(rowsize)] for j in range(colsize)]
payload = { 'weights' : weights }

payload = json.dumps(payload).encode('utf-8')

res = json.loads(payload)

HOST, PORT = 'ec2-3-18-131-23.us-east-2.compute.amazonaws.com', 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
	sock.bind((HOST, PORT))
	sock.listen(10)
	client, addr = sock.accept()

	print(client.recv(1024))

	client.send(payload)
except:
	print("SOMETHING WENT WRONG")
finally:
	sock.close()
