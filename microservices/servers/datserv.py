import json
import random
import socket
import ssl

n_nodes = 10
n_feats = 10

feats = [[int(random.random() * 10) for i in range(n_feats)] for j in range(n_nodes)]
adj_list = []

def create_adj_list():
	for i in range(n_nodes):
		n_edges = int(random.random() * n_nodes)
		edge_list = set([int(random.random() * n_nodes) for x in range(n_edges)])
		edge_list = list(edge_list)
		edge_list.sort()
		adj_list.append(edge_list)

create_adj_list()

print(feats)
print(adj_list)

def sumVecs(a, b):
	return [x+y for x, y in zip(a, b)]




HOST, PORT = 'ec2-3-18-131-23.us-east-2.compute.amazonaws.com', 65432

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
	sock.bind((HOST, PORT))
	sock.listen(10)
	client, addr = sock.accept()

	print(client.recv(1024))

	buf = []
	for n in range(n_nodes):
		aggFeats = [0 for f in range(n_feats)]
		edges = adj_list[n]
		for e in edges:
			aggFeats = sumVecs(aggFeats, feats[e])
		buf.append(aggFeats)

	payload = { 'feats' : buf }
	payload = json.dumps(payload).encode('utf-8')

	client.send(payload)
except:
	print("SOMETHING WENT WRONG")
finally:
	sock.close()
