---
layout: post
title: "Image processing using serverless architecture"
author: Victor
date: '2020-02-24'
category: Architecture
summary: How to configure AWS lambda to build a serverless image processing application
thumbnail: serverless.png
published: true
language: EN
tag: EN
---

# Introduction
Most computer vision applications need a server for hosting. This is a classic step that every developer encounters as the application is ready to be shared, which increases complexity and add an extra cost for the production environment.

Cloud solutions make easy to deploy backend applications. Load balancing, security, deployment, monitoring and many other topics are not a problem anymore. Serverless solutions, like Serverless Framework, Google Cloud or AWS Lambda rely on their own platforms, which are configured using proprietary tools. However, it may turn out complicated whenever your application is outside the canonical uses. It was our case during the IA Flash project, for **sending binary data to our IA application**. We managed to solve this problem as you can see in our portfolio [page](https://iaflash.fr). In this article we wanted to explain the main steps to overcome this difficulty.

We created a simple project to show how to deploy a lambda function that returns the shape and size in bytes of an image by passing binary files. The simplicity of the project makes that the reader can take our repository as a template to deploy Lambda API and then use more complicated functions.

Another relevant solution is to trigger lambda execution by uploading an S3 image, impossible for us because of the sensitive aspect of our data.

In a nutshell, we tackle the problem of posting to aws-lambda those kind of requests :

```bash
curl "https://2qhuxkow20.execute-api.eu-west-1.amazonaws.com/Prod/shape" -H "Content-Type: image/jpg" --data-binary "@test/cat.jpg"`
```

And even more interesting, form data requests which allows the possibility of sending **multiple images** in the same request ;

```bash
`curl "https://7w26u0d56g.execute-api.eu-west-1.amazonaws.com/Prod/shape"  -H "Content-Type: multipart/form-data" -F "image=@test/cat.jpg"
```

## Code
The code is available in our GitHub : https://github.com/victorjourne/aws-lambda-cv. One branch is dedicated to post a binary file (data-binary) and the other to submit a list of binary files (form-data).

This post will also focus on setting up a serverless architecture for image processing. This would be deployed in the cloud using AWS lambda service.(SAM).

The serverless architecture consists of deploying **only a function** that can be easily used by multiple machines depending on users demand. This approach removes the scalability problem from the developer because the provider has the control of the number of resources that would be deployed. This approach also reduces cost for applications that are not used extensively, because most providers charge depending on the number of executions. For the case of AWS Lambda, the service has **no charge** until 3 millions of requests.

A typical use case where we can use serverless function is for image processing. However is not always simple to handle encodings and sending image through body requests and gets even harder when you try to send multiple images on the same requests as an attachment.

## Send multipart data
We will develop only the multipart case, when the client uploads a list of binary files.
On the lambda side, the request content is retrieved with:

```python
# lambda_function.py
def lambda_handler(event, context):

    res = list()
    assert event.get('httpMethod') == 'POST'
    try :
           event['body'] = base64.b64decode(event['body'])
```

Finally, we get back the binary images by calling the multipart decoder from *request-toolbelt* python package.

```python
multipart_data = decoder.MultipartDecoder(event['body'], content_type)
for part in multipart_data.parts:     
   img_io = io.BytesIO(part.content)
   img_io.seek(0)
   img = Image.open(img_io)
```

# Deployment

Aws provides a command-line interface to build, test, debug, and deploy Serverless applications. This tool is an extension of *AWS CloudFormation*, which manages the deployment of our cloud resources, defined inside a text file.

ApiGateway, the brick between the post request and the lambda function must be configured (in the template.yaml) so as to support and *base64* encode the *multipart/form-data content*.

```yaml
# template.yaml
BinaryMediaTypes:
   - multipart~1form-data
```

Then, on the swagger.yaml,  the apigateway integration should be type as a proxy (aws_proxy). In this case, requests will be proxied to our lambda and the content will be placed inside the event inside event object.


```yaml
# swagger.yaml
x-amazon-apigateway-binary-media-types:
    - '*/*'
x-amazon-apigateway-integration:
    type: "aws_proxy"
    httpMethod: "POST"
    uri: "arn:aws:apigateway:eu-west-1:lambda:path/2015-03-31/functions/arn:aws:lambda:eu-west-1:016363657960:function:ToyFunctionForm/invocations"
```

# Tests

Deploying to the cloud often implies some waiting time between each new version of the application. This time can be reduced by using local test functions. For our case, we used the pytest library, which has nice liting features and also has the possibility of centralizing test inputs in a single configuration file called `conftest.py`. Thus, the python tests replicate the apigateway behaviour.

In addition to this library we also used *MultipartEncoder* to test complex parameters for http requests, such as encoding multi part images. In the following blocks, we show how to test our lambda function:

```python
def test_handler(apigateway_event):
    resp = lambda_handler(apigateway_event('shape'), None)
    body =     resp['body']
    print(body)
    assert resp['statusCode'] == 200
    assert eval(body)== [[600, 400, 3]], "shape of image is : %s"% eval(body)
    resp = lambda_handler(apigateway_event('nbytes'), None)
    body = resp['body'] print(body)
    assert resp['statusCode'] == 200   
    assert eval(body)== [720], "size of of image is : %sk bytes "% eval(body)
```

# Conclusions
Command line wrapper such as *AWS SAM CLI* makes deploying AWS applications very simple. A plain text defines the parameters of the services, which allows reproducibility of the deployment. In addition this process can be automated and coupled with CI/CD pipelines, for example deploying a lambda function whenever a new version of an application is pushed.

Automating the deployment process with aws sam cli is a complete relief because:
Deploying your function is as simple as a git push
Development environment is the same as the production environment
Tests are a guarantee that your code will work
This comes also with serverless advantages: You don’t have to worry about **server maintenance and scalability**!
In addition most serverless providers have very interesting economic prices: AWS for example gives 3 millions of requests by month for free if your function uses less than 128 ram memory.
