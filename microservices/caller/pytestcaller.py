import botocore.response as br
import boto3
import json
import socket

# 

payload = """{
	"ip" : "ec2-3-18-131-23.us-east-2.compute.amazonaws.com",
	"wport" : 12345,
	"fport" : 65432
	} """


client = boto3.client('lambda')
response = client.invoke(
	FunctionName="pytest",
	InvocationType="RequestResponse",
	Payload=payload
	)

result = response
result = result["Payload"].read()

z = json.loads(result)['result']

for vec in z:
	print(vec)
