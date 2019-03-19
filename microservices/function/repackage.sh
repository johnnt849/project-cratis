#! /bin/bash

rm myfunction.zip
zip -r9 myfunction.zip myfunction.py

aws lambda update-function-code --function-name pytest --zip-file fileb://myfunction.zip
