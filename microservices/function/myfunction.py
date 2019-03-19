import json
import socket

# the lambda test function:
# receives an ip and port of "data servers" then
# reads data, processes it and returns

def dot(a, b):
	result  = [x*y for x,y in zip(a, b)]
	return sum(result)

def handler(event, context):
	server_ip = event['ip']
	wport = event['wport']
	fport = event['fport']

	weights = None
	wsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try:
		wsock.connect((server_ip, wport))
		wsock.send(b'weights')

		weights = wsock.recv(50000)
		weights = json.loads(weights)['weights']

	except:
		print("Coudln't get weights")
	finally:
		wsock.close()


	feats = None
	fsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try:
		fsock.connect((server_ip, fport))
		fsock.send(b'feats')

		feats = fsock.recv(500000)
		feats = json.loads(feats)['feats']

	except:
		print("Couldn't get features")
	finally:
		fsock.close()

	result = []
	for fvec in feats:
		result.append([dot(fvec, wvec) for wvec in weights])

	return {
		'result' : result
	}
